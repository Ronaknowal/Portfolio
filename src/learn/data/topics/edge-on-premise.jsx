import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const edgeOnPremise = {
  title: "Edge & On-Premise Deployment Architectures",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Most writing about LLM deployment assumes public cloud — hyperscaler GPUs, multi-tenant endpoints, commodity APIs billed by the token. The mental model is sensible for the majority of use cases: rent capacity, call an endpoint, pay per query. But a real and growing fraction of the market runs LLMs inside customer datacenters, on-device, or on edge hardware where that model breaks down entirely. The requirements diverge sharply — compliance, latency, offline operation, data residency — and the architecture follows the requirements. This topic covers what changes when the cloud assumption goes away and what the resulting deployment patterns actually look like.
      </Prose>

      <H2>Why leave the cloud</H2>

      <Prose>
        The reasons sort cleanly into three categories, and they are worth distinguishing because they impose different constraints and point toward different solutions.
      </Prose>

      <Prose>
        Data sovereignty is the most common forcing function in regulated industries. Healthcare, finance, and defense operate under legal frameworks — HIPAA, SOC 2, FedRAMP, export control regimes — where data leaving the customer's infrastructure is not a risk to manage but a violation to avoid. Cloud APIs, however well-secured, involve data transiting third-party infrastructure. For a hospital system asking an LLM to summarize patient records, or a defense contractor using one to parse classified documents, the legal calculus makes cloud deployment impossible before the first performance benchmark is run. On-premise is not a preference in these environments. It is a compliance boundary.
      </Prose>

      <Prose>
        Latency is the second category, and it is distinct from the sovereign data problem in that it is purely physical. A round-trip to the nearest cloud datacenter takes somewhere between 20ms and 150ms depending on geography and network conditions. For many applications that is irrelevant. For robotics, AR/VR, real-time translation, in-vehicle assistants, and industrial control systems, it is the difference between a usable product and an unusable one. A robot arm that needs to process a vision query before deciding where to move cannot wait 80ms for a cloud response. The only solution is inference local to the device, which means the model must run on hardware that is physically close to or part of the system it is serving.
      </Prose>

      <Prose>
        Offline operation is the third, and the most absolute. Ships at sea, aircraft in flight, field deployments in areas without reliable connectivity, classified environments with no external network — in all of these the cloud endpoint is simply unreachable. The model must run locally or not at all. This is a different problem than latency optimization; no amount of network engineering solves it. The system must be self-contained.
      </Prose>

      <H2>The on-premise tier</H2>

      <Prose>
        The most common non-cloud case is also the closest to cloud deployment in architecture. A customer runs the full LLM serving stack inside their own datacenter, on GPUs they own or lease. The software — <Code>vLLM</Code>, <Code>TGI</Code>, <Code>SGLang</Code>, or a vendor-supplied serving framework — is the same software used in cloud deployments. The operational environment is what changes.
      </Prose>

      <Prose>
        The differences accumulate. There is no elastic autoscaling: GPU capacity is fixed, and when demand exceeds supply the system queues requests rather than spinning up new instances. The customer owns monitoring, model updates, and the entire operational lifecycle — there is no managed control plane pushing patches or alerting on regressions. The deployment is typically single-tenant, so the scheduling complexity around fairness and quota management that occupies cloud serving systems largely disappears. The environment is often airgapped from the model provider's infrastructure, which means updates are batched, manual, and auditable rather than continuous.
      </Prose>

      <Prose>
        This is the architecture of most enterprise LLM deployments in 2025: NVIDIA Enterprise AI, AWS Outposts with GPU instances, Dell APEX, HPE Private Cloud AI. The pattern is cloud LLM serving minus multi-tenancy and elasticity, plus the full operational burden of running infrastructure yourself. The tradeoff is data sovereignty, predictable cost at scale, and the ability to operate without an external dependency. For large organizations with existing datacenter operations and compliance requirements, the tradeoff often clears easily.
      </Prose>

      <H2>Edge deployment — the hardware constraint</H2>

      <Prose>
        Edge is a different regime. Where on-premise datacenter hardware operates in the same power and thermal envelope as cloud servers — hundreds of watts per GPU, thousands of watts per rack — edge hardware has power budgets measured in tens to low hundreds of watts. That constraint propagates directly into what models can run. A 70B parameter model in FP16 requires roughly 140GB of memory; the NVIDIA H100 with 80GB HBM requires two units minimum, consumes 700W each, and costs roughly $30,000 per card. Edge hardware has none of those characteristics. The constraint is not artificial. It is the physics of putting inference inside a vehicle, a phone, a factory floor unit, or a portable device.
      </Prose>

      <CodeBlock>
{`tier                     example hardware                     model class running well
cloud datacenter         NVIDIA H100 (700W, 80GB HBM)         Frontier (100B+ params)
enterprise datacenter    NVIDIA L40S (350W, 48GB)             7-70B models
workstation              RTX 4090 (450W, 24GB)                7-30B quantized
edge server              Jetson AGX Orin (60W, 64GB)          3-8B
laptop                   M3 Max / M4 Pro                      3-8B quantized
phone / on-device        mobile NPU / Neural Engine           1-4B heavily quantized`}
      </CodeBlock>

      <Prose>
        The tier boundaries are not sharp — a Jetson AGX Orin at 60W and 64GB of unified memory will run an 8B model quite comfortably, while a power-constrained laptop will struggle with the same model at full precision. What matters is the general shape: as power budget falls by an order of magnitude, the largest model that runs well falls by roughly the same factor, and quantization becomes the primary tool for staying within the envelope.
      </Prose>

      <H2>Quantization as the enabler</H2>

      <Prose>
        Edge deployment is mostly a story of aggressive quantization. The intuition is simple: a floating-point weight in FP16 takes 2 bytes; in INT4 it takes half a byte; and a model that fits in memory runs while one that does not runs nowhere. A Llama 3 8B model at FP16 precision occupies roughly 16GB of memory. In 4-bit quantization it fits in 4GB — comfortably within the memory envelope of a consumer GPU, an M-series laptop, or a high-end phone. That factor of four is the difference between "edge is impossible" and "edge is routine."
      </Prose>

      <Prose>
        The quality cost of aggressive quantization is real but smaller than intuition suggests. Weight-only quantization methods — AWQ and GPTQ being the most widely deployed — quantize weights while preserving activations at higher precision, which recovers most of the quality lost by naive rounding. Mixed-precision schemes like SmoothQuant and the FP4/FP8 formats supported by recent NVIDIA hardware push further while maintaining differentiable training paths. Measured against standard benchmarks, 4-bit quantization of a well-calibrated model costs roughly 1-3 percentage points on broad evaluation suites, and often less than 1 percentage point on narrow, domain-specific tasks. For most deployment use cases that is an acceptable tradeoff for a 4x memory reduction. The PEFT and Model Optimization track covers quantization methodology in depth; here the point is simply that quantization is what makes the hardware table above possible rather than theoretical.
      </Prose>

      <H2>On-device — the frontier cases</H2>

      <Prose>
        The most aggressive end of the edge spectrum is on-device inference on consumer hardware — phones, tablets, and laptops with no discrete GPU at all. This has moved from a research curiosity to a shipping product class in roughly two years. Apple Intelligence ships 3B-parameter models running entirely on-device via Apple Silicon's Neural Engine, handling summarization, writing assistance, and contextual actions without any network call. Google's Gemini Nano runs on Pixel phones for on-device tasks including summarization and Smart Reply. Microsoft ships Phi-series models tuned for Snapdragon X NPUs, targeting the PC form factor. These are not frontier models — they are narrowly capable, task-targeted systems optimized for a specific set of high-frequency user interactions. But they are adequate for a useful fraction of user tasks, and the inference is effectively free: no API call, no network round-trip, no latency, no per-query cost.
      </Prose>

      <Prose>
        The architecture that makes on-device practical is not just small models. It is the combination of small models, aggressive quantization, hardware-specific kernel optimization (Core ML, QNN SDK, ExecuTorch), and careful task scoping. A 3B model asked to summarize a notification or autocomplete a sentence performs credibly. The same model asked to reason through a complex multi-step problem does not. The deployment pattern that resolves this is hybrid routing: on-device for fast, common, latency-sensitive tasks; cloud fallback for capability-bound queries that the local model cannot handle reliably. The classifier that decides which path a request takes is itself a small on-device model — cheap to run, fast to respond, and able to make the routing decision before the user notices any latency.
      </Prose>

      <H3>The operational differences</H3>

      <Prose>
        Running LLMs outside the cloud changes what production operations look like in ways that go beyond the hardware constraints. Updates are rare and batched rather than continuous. In cloud serving, a new model version can be rolled out to a fleet in hours and rolled back in minutes. On a customer-operated server or an on-device deployment, a new model version ships quarterly or on a hardware-vendor release cycle, and rollback requires the same manual process. This changes the testing and validation requirements significantly — the cost of a bad deployment is weeks of exposure, not hours. Monitoring must function offline or in degraded network conditions. Cloud observability assumes constant metric export and centralized dashboards; edge systems must buffer telemetry locally, batch uploads opportunistically, and support debugging through on-device logs rather than real-time traces.
      </Prose>

      <Prose>
        The security model shifts in a less obvious way. In cloud deployment, the model weights live on infrastructure the operator controls; the threat model is API abuse, prompt injection, and data leakage through the serving layer. In on-device or customer-premise deployment, the weights are distributed to untrusted hardware. The model binary is now user-accessible data in the same sense that any installed application is. Signing, verification, and rollback become first-class engineering requirements — not for the model's safety properties but for the software-update integrity of the deployment. Finally, thermal and power constraints become product-defining. A model that runs at 30 tokens per second for the first minute but throttles to 5 tokens per second under sustained load due to thermal management is a different user experience than a model that sustains 15 tokens per second indefinitely. That distinction does not exist in cloud serving, where thermal management is the datacenter operator's problem. On edge hardware, it is the application developer's problem.
      </Prose>

      <H2>Hybrid architectures — the emerging default</H2>

      <Prose>
        Pure edge and pure cloud are both endpoints of a spectrum; most real deployments in production today sit somewhere between. The pattern that is emerging as a default for consumer-facing applications combines a small capable model on-device for latency-sensitive and offline work with a large model in the cloud for capability-bound queries, and a routing layer that decides which path each request takes. The routing decision happens locally — fast, cheap, and available offline — and the cloud is invoked only when the local model is likely to fall short. From the user's perspective, the system responds instantly for common tasks and reaches for heavier capability only when the task genuinely requires it. The seam between on-device and cloud is meant to be invisible.
      </Prose>

      <Callout accent="gold">
        The future of most LLM deployment is hybrid — a small capable model on-device, a large one in the cloud, and routing between them that looks invisible to the user.
      </Callout>

      <Prose>
        The practical engineering challenges in hybrid architectures concentrate in the router. A router that is too aggressive about sending requests to the cloud degrades the latency and cost properties that justified on-device inference in the first place. A router that is too conservative produces visible quality failures on tasks the local model cannot handle. Calibrating the boundary requires careful analysis of the task distribution, the capability gap between the on-device and cloud models, and the latency tolerance of the application. This is not a solved problem — it is one of the active engineering areas in applied LLM deployment — but the tooling is maturing quickly as the major on-device frameworks have all added routing primitives.
      </Prose>

      <Prose>
        Edge and on-premise deployment are where the economics and requirements of LLM serving stop being a cost-optimization problem and start being a software-distribution problem. The fixed capacity, the batched update cycle, the offline monitoring, the model signing requirements — all of these are the vocabulary of software shipped to endpoints, not of services operated in a datacenter. The final topic in this section — multi-region and global inference infrastructure — addresses the opposite extreme: what happens when you need LLM serving on every continent simultaneously, and the challenge is not distribution to constrained hardware but coordination across a globe-spanning fleet.
      </Prose>
    </div>
  ),
};

export default edgeOnPremise;
