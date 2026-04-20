import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const streamingSSE = {
  title: "Streaming & Server-Sent Events (SSE)",
  readTime: "9 min",
  content: () => (
    <div>
      <Prose>
        Almost every production LLM API streams. Rather than waiting for the model to finish generating a full response before sending anything, the server flushes tokens to the client as they are produced. The user sees text appearing word by word. Mechanically, this is Server-Sent Events: a long-lived HTTP response where the server writes incremental payloads and the client reads them in real time. The wire format is three lines of spec. The operational reality is that a surprising fraction of standard HTTP infrastructure quietly breaks it, and the bugs only surface in production.
      </Prose>

      <H2>Why streaming matters</H2>

      <Prose>
        The latency metric that users actually feel is time-to-first-token — how long before anything appears on screen. End-to-end latency, the time until the response is complete, matters for scripted pipelines but not for the human reading the output. A ten-second generation that starts appearing after 400 ms feels fast. A three-second generation that waits two full seconds before the first character appears feels broken. Streaming collapses the perceptible latency to the time-to-first-token, which for most production systems sits in the 200–600 ms range regardless of how long the full completion takes.
      </Prose>

      <Prose>
        The second reason is early cancellation. If a model starts generating a response that is clearly wrong — wrong tone, wrong language, the wrong tool call — the user can abort. Without streaming, the client waits for the full generation and then discards it. With streaming, the client sees the error after the first few tokens and closes the connection. The server, if it is written correctly, detects the disconnect and cancels the in-flight generation immediately. The tokens that would have been generated are never generated. At scale, across thousands of concurrent users, the compute saved is substantial.
      </Prose>

      <H2>The SSE wire format</H2>

      <Prose>
        SSE is a plain-text protocol over HTTP. The response has <Code>Content-Type: text/event-stream</Code>. The body is a sequence of <Code>data:</Code> lines, each followed by two newlines. An optional <Code>event:</Code> prefix names the event type; without it, the client sees a generic <Code>message</Code> event. The payload of each <Code>data:</Code> line is whatever the server wants to send — for LLM APIs it is almost always JSON. A final sentinel value, typically <Code>[DONE]</Code>, signals the end of the stream.
      </Prose>

      <CodeBlock>
{`HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" there"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]

`}
      </CodeBlock>

      <Prose>
        The client parses each <Code>data:</Code> line as it arrives, extracts the delta, and appends it to the displayed text. The double newline between events is load-bearing — it is what separates one event from the next. Everything else is convention. OpenAI's format is the de facto standard: a <Code>choices</Code> array with a single element, a <Code>delta</Code> object containing the incremental <Code>content</Code>, and a <Code>finish_reason</Code> field on the final chunk to tell the client why generation stopped.
      </Prose>

      <H3>Implementing an SSE endpoint</H3>

      <CodeBlock language="python">
{`from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    async def event_stream():
        async for token in llm.generate_stream(request.prompt):
            payload = {
                "choices": [{"delta": {"content": token}}],
            }
            yield f"data: {json.dumps(payload)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )`}
      </CodeBlock>

      <Prose>
        The critical detail is the double newline at the end of each <Code>yield</Code>. Omit one and the client receives an event it cannot parse. FastAPI's <Code>StreamingResponse</Code> calls the async generator and flushes each yielded chunk immediately — it does not buffer. That flush behavior is correct and is exactly what you want. The problem is everything between the FastAPI process and the user's browser.
      </Prose>

      <H2>Why off-the-shelf proxies break it</H2>

      <Prose>
        SSE requires that every flushed byte reaches the client without waiting for more bytes to accumulate. Most standard HTTP components are not built with that assumption. They are built to optimize throughput, which means buffering. A load balancer that buffers responses collects all the tokens the server sends, holds them, and then delivers them in a single burst when the connection closes or the buffer fills. From the user's perspective, the page is blank for ten seconds and then the entire response appears at once. The streaming behavior is completely invisible. The server is streaming correctly; the infrastructure is erasing it.
      </Prose>

      <Prose>
        The failure modes form a recognizable list. GCP's HTTP(S) Load Balancer buffers streaming responses by default; the fix is to set the backend service's <Code>enableCDN</Code> to false and use a backend bucket with streaming enabled, or route through a TCP proxy instead. Nginx with default settings applies <Code>proxy_buffering on</Code>; add <Code>proxy_buffering off</Code> and <Code>X-Accel-Buffering: no</Code> on the streaming route. Compression middleware — gzip, brotli — collects bytes before emitting compressed blocks, which completely destroys flush semantics; disable compression for <Code>text/event-stream</Code> responses. Cloudflare has historically buffered streaming responses on certain plan tiers. Envoy needs <Code>response_buffer_limit_bytes: 0</Code>. The pattern is the same everywhere: buffering is an optimization that improves performance for normal responses and catastrophically breaks streaming ones, and it is usually on by default. Teams get streaming working on localhost, deploy behind a production LB they did not configure for streaming, and discover the problem from user complaints, not from their test suite.
      </Prose>

      <H3>Alternatives — WebSockets and gRPC streaming</H3>

      <Prose>
        SSE is the default choice for browser-facing LLM APIs because it is plain HTTP — no protocol upgrade, no special client library, works through any HTTP/1.1 connection. The browser's <Code>EventSource</Code> API handles it natively. The limitation is that SSE is unidirectional: the server sends, the client reads. For standard chat applications this is fine. The user sends a message, the model streams the response, the user sends another message.
      </Prose>

      <Prose>
        WebSockets add bidirectional streaming. This becomes relevant in agent frameworks where the client needs to interrupt the model mid-generation — supplying a tool result, injecting a new instruction, or canceling a specific branch of a tree search. With SSE, these interactions require opening a new HTTP request; with WebSockets, the client and server share a persistent channel and can interleave messages freely. gRPC streaming offers the same bidirectional capability with strong type contracts and better observability tooling, which makes it attractive for server-to-server traffic between inference services and orchestration layers. For anything touching a browser, SSE is simpler and sufficient; for agent frameworks where the client is itself a program running tools during generation, WebSockets or bidirectional gRPC earn their added complexity.
      </Prose>

      <H2>Cancellation</H2>

      <Prose>
        Streaming without cancellation is half the feature. When a client closes the connection — user pressed stop, navigated away, network dropped — the server needs to know. In FastAPI, an <Code>asyncio.CancelledError</Code> propagates into the async generator when the client disconnects; wrap the generation loop in a try/except and pass a cancellation signal to the inference engine. Node's <Code>res.on('close', ...)</Code> fires on disconnect. Go's <Code>net/http</Code> surfaces a cancelled context on <Code>req.Context()</Code>. The signal is available in every major serving framework. The question is whether the inference engine downstream actually honors it.
      </Prose>

      <Prose>
        vLLM, SGLang, and TGI all support cancel-on-disconnect; they stop the generation and release the KV cache slot when the request is aborted. Naive inference wrappers that do not plumb the cancel signal will keep generating until the sequence is complete, burning compute and KV cache memory for a response no client will ever read. At low traffic this is invisible. At high traffic with a non-trivial user abort rate — which any production chat product has — the wasted compute adds up quickly. Cancellation is not a quality-of-life feature. It is a resource efficiency feature.
      </Prose>

      <H3>Partial response handling</H3>

      <Prose>
        The stream does not always end cleanly. A safety filter triggers at token 200 of a 300-token generation. A timeout fires. The inference node crashes. The client has partial text and needs to decide what to do with it. The options sit on a spectrum of user-friendliness. Silently truncating — displaying whatever arrived and pretending it is complete — misleads the user into treating a partial response as a whole one. Emitting an explicit error event lets the client show a "response cut short" indicator so the user knows to retry. Retrying from the truncation point requires the client to resubmit with the accumulated context as the new prompt, which works but adds latency and complexity.
      </Prose>

      <Prose>
        The industry convention is to emit a <Code>finish_reason</Code> field on the final chunk of every stream. A clean completion sends <Code>finish_reason: "stop"</Code>. A generation truncated at the token limit sends <Code>"length"</Code>. A safety filter sends <Code>"content_filter"</Code>. An error mid-generation sends <Code>"error"</Code>, often accompanied by an <Code>error</Code> field with a code and message. The client inspects <Code>finish_reason</Code> on the last chunk and branches accordingly. A client that only handles the happy path — final token received — will silently swallow the other three cases and produce a confusing UX.
      </Prose>

      <Callout accent="gold">
        A well-behaved streaming client handles three states: final token received, connection died mid-stream, explicit early termination with reason. Miss any of the three and the UX suffers.
      </Callout>

      <Prose>
        Streaming is the defining UX choice of modern LLM products. The difference between a product that feels alive and one that feels like a web form waiting to load is almost entirely in whether tokens appear immediately. Getting the plumbing right — flush semantics across the full proxy chain, cancellation that reaches the inference engine, partial-response handling that communicates honestly to the user — is below-the-line infrastructure work. It is invisible when correct and immediately felt when wrong. The next topic turns to cost optimization at the organizational scale, where the unit economics of tokens-per-dollar become the central engineering concern.
      </Prose>
    </div>
  ),
};

export default streamingSSE;
