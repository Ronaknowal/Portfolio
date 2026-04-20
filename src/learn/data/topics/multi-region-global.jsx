import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const multiRegionGlobal = {
  title: "Multi-Region & Global Inference Infrastructure",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Serving LLMs globally means running the same service in multiple regions, across
        continents, with consistent product behavior. At the hyperscaler level — OpenAI,
        Anthropic, Google, AWS Bedrock — this collapses most of the interesting
        distributed-systems problems into a single operational surface. GPU capacity is
        unevenly distributed across regions. User traffic isn't uniform across time zones.
        Data residency regulations differ by jurisdiction in ways that are legally binding,
        not advisory. And the largest models won't fit in some datacenters at all, because
        the hardware simply isn't there yet. Getting a token from the right model to the
        right user with the right latency, under those constraints, is not a solved problem.
        It is a continuous engineering negotiation.
      </Prose>

      <H2>Why regions matter</H2>

      <Prose>
        Three distinct drivers shape global inference topology, and they pull in different
        directions. Understanding each one clarifies why the resulting architecture looks
        the way it does.
      </Prose>

      <Prose>
        The first is latency. A request round-tripping from Sydney to us-east-1 adds
        roughly 200ms of network time before a single inference FLOP is spent. For
        interactive chat, that is noticeable — users perceive round-trip delays above
        150ms. For streaming, it still matters: the first token doesn't arrive until the
        network round-trip completes, so TTFT for a Sydney user hitting a Virginia endpoint
        is structurally degraded regardless of how fast the model generates. Putting
        inference compute closer to users is the most direct fix for this, and every
        hyperscaler has deployed to APAC-east, EU-west, and US-west precisely because
        latency is felt before the model speaks a word.
      </Prose>

      <Prose>
        The second is data residency. GDPR requires that EU personal data stay within the
        EU. Sector-specific rules go further: a Canadian healthcare operator's patient
        prompts cannot leave Canada; a US defense contractor's requests may not leave
        FedRAMP-authorized infrastructure. Customer contracts frequently mirror these
        requirements even where regulation is silent. The practical consequence is a hard
        constraint on request routing: an EU user's prompt, response, and any cached state
        derived from it must stay within an EU region, regardless of where spare capacity
        sits. No amount of latency or cost pressure overrides this. Residency requirements
        are the first filter applied to any routing decision, not the last.
      </Prose>

      <Prose>
        The third is GPU supply. Even a hyperscaler cannot put H100s everywhere. GPU supply
        is constrained, arrives in uneven tranches, and varies by region based on power
        infrastructure, datacenter buildout, and supply-chain allocation. A frontier lab
        running Llama 3 70B and a 405B parameter model simultaneously may have the 70B
        available in seven regions and the 405B in two. Users in the regions without 405B
        capacity either get the smaller model, wait for cross-region routing, or get an
        error. Hardware availability is an operational constraint that is visible to users
        as a capability gap, which is why it gets managed as carefully as latency.
      </Prose>

      <H2>Regional topology</H2>

      <Prose>
        A typical global LLM product partitions the world into roughly five to ten regions:
        us-east, us-west, eu-west, eu-central, apac-east, apac-south, and a few others
        depending on the provider's customer geography. Each region carries a full serving
        stack — routers, autoscalers, inference workers, KV cache, safety classifiers,
        observability pipelines. From the outside, every region looks like a self-contained
        copy of the product. Users are directed to their nearest or regulatorily appropriate
        region via DNS or a global traffic manager, and in the normal case never know which
        region served them.
      </Prose>

      <Prose>
        The split between what is shared across regions and what is regional-only is not
        arbitrary. Model weights are shared — they are immutable artifacts replicated to
        every region that needs them, but they are not user data and can move freely across
        borders. Safety classifiers and policy documents follow the same logic. What stays
        regional: user data, KV cache state, request logs, per-region fleet telemetry, and
        anything downstream from a user's prompt. That clean split — weights travel, user
        data does not — is the architectural expression of the data residency constraint.
        The serving infrastructure enforces it structurally rather than procedurally, which
        is the right approach when the penalty for violating it is a GDPR fine.
      </Prose>

      <H3>Request routing — anycast and geo-DNS</H3>

      <Prose>
        Two mechanisms dominate global request routing. Geo-DNS resolves
        <Code>api.example.com</Code> differently depending on where the DNS query
        originates, directing users to their nearest regional endpoint at the DNS layer.
        Anycast takes a different approach: the same IP address is advertised from multiple
        regions, and BGP routes each packet to whichever announced origin is topologically
        closest. Anycast is simpler from the client's perspective — there is a single
        endpoint, and the network handles region selection transparently. It is more complex
        to operate: BGP convergence, region-specific health propagation, and anycast prefix
        management all require careful coordination. Geo-DNS is easier to operate but
        creates issues for mobile users and VPN'd traffic, where the DNS resolver's
        location differs from the user's actual location.
      </Prose>

      <Prose>
        Neither mechanism alone handles the full constraint set. Latency-closest is not
        always the right region — data residency may prohibit it. The routing layer needs
        to enforce hard residency constraints before it optimizes for anything else.
      </Prose>

      <CodeBlock language="python">
{`# Region selection considering both proximity and data residency

def select_region(request, regions):
    # Hard constraints first — residency wins over latency
    allowed = [r for r in regions if r.allows_data_from(request.user_region)]
    if not allowed:
        raise NoEligibleRegionError(request.user_region)

    # Among allowed regions, pick the one with lowest expected latency
    allowed.sort(key=lambda r: (
        r.network_latency_to(request.user_region),
        r.current_load,
    ))
    return allowed[0]`}
      </CodeBlock>

      <H2>Data residency in practice</H2>

      <Prose>
        Residency is the hardest constraint in global inference, and it cascades further
        through the stack than most engineers expect the first time they hit it. The obvious
        consequence is that a regulated customer's prompts and responses must stay in-region.
        The less obvious consequences stack on top. KV cache state is derived from the
        prompt — it is effectively a compressed representation of the user's input — so it
        is region-local and cannot be shared cross-region even for identical prompts from
        different users. Request logs stay in-region. Per-user metrics may need to filter
        PII before any aggregation crosses a region boundary, which means the observability
        pipeline has different rules for regulated versus non-regulated traffic. Even the
        act of routing to a cheaper region's spare capacity is unavailable for residency-
        constrained requests, regardless of the cost savings on offer.
      </Prose>

      <Prose>
        Model weights are usually exempted — they are not user data, and most jurisdictions
        do not restrict their movement. Some enterprise contracts go further and restrict
        even weight transfer, on the theory that knowing which model version a customer is
        running constitutes a confidential business detail. This is rare but non-zero in
        heavily regulated verticals. The practical effect is that a provider with
        particularly strict contract terms may need to maintain separate model deployment
        pipelines per customer cohort, which multiplies operational surface considerably.
      </Prose>

      <H3>Failover and capacity pooling</H3>

      <Prose>
        Regional outages are infrequent but not rare enough to ignore. When a region goes
        down, the serving architecture faces a branching decision. For residency-constrained
        traffic, the answer is often forced: there may be no eligible alternative region,
        and the correct response is a graceful failure rather than a residency violation.
        Regulated customers typically accept this tradeoff explicitly — the contract says
        "data stays in EU-west" and the implicit corollary is "if EU-west is unavailable,
        service is unavailable." Some providers add a standby region within the same
        jurisdiction (EU-central as a failover for EU-west), which resolves the availability
        problem without creating a residency one, at the cost of maintaining two regional
        stacks for what is often a small customer segment.
      </Prose>

      <Prose>
        For non-residency-constrained traffic, cross-region failover is standard. The
        global traffic manager detects the failing region's health check and shifts traffic
        to the next-nearest healthy region. Users see increased latency; they do not see
        errors. Capacity sharing — allowing one region's overflow to route into an
        under-utilized nearby region — is a related technique used during traffic spikes
        rather than outages. The same residency filters apply: overflow can only route to
        regions that are eligible for the originating traffic. The practical result is that
        capacity pooling helps non-regulated traffic considerably and helps regulated traffic
        not at all, which is another reason the regulated-customer stack gets treated as
        operationally distinct.
      </Prose>

      <H3>The model-deployment problem</H3>

      <Prose>
        Every region needs to run the same model version — or at minimum, a controlled set
        of versions. Rolling out a new model to all regions takes days in practice. Weights
        must transfer (a 70B model at BF16 is 140GB; a 405B is 810GB), regional caches
        need warm-up time, eval validation must pass in each region before traffic shifts,
        and the traffic shift itself is typically gradual — 1%, 5%, 25%, 100% — with
        automated rollback triggers keyed to quality metrics. The entire process for a major
        model version, done safely, is a multi-day operation with active monitoring at
        every stage.
      </Prose>

      <Prose>
        Partial rollouts — a new version in two regions while the old version runs in
        five — create "region-dependent behavior" bugs that are disproportionately hard to
        debug. A user testing from London hits the new model; a user testing from Tokyo
        hits the old one; their reported behaviors are contradictory; neither support ticket
        mentions the region. Production practice at shops that have hit this class of bug
        converges on one-region-at-a-time rollouts with automated quality eval as a rollback
        trigger, accepting the slower rollout cadence as the cost of debuggable production
        behavior. The alternative — fast parallel rollout across all regions — is faster
        until it produces an incident, at which point it produces a much worse incident.
      </Prose>

      <H2>Capacity arbitrage</H2>

      <Prose>
        GPU availability and pricing vary significantly across regions. US regions often run
        at high utilization with constrained H100 supply; newer regions — EU-central, some
        APAC zones — may carry spare capacity at lower effective cost. For workloads that
        are not latency-critical and carry no residency constraint — batch inference, offline
        evaluation, synthetic data generation, embedding precomputation — there is no reason
        to serve them from the most congested region. Sophisticated operators route these
        flexible workloads to whichever region has spare supply at lower cost this week,
        treating inter-region capacity differences as an arbitrage opportunity rather than
        an inconvenience. The savings are not marginal: shifting 20-40% of a compute bill
        by routing batch work to cheaper regions is achievable once the routing and
        monitoring infrastructure exists to do it safely.
      </Prose>

      <Prose>
        The cost is operational complexity. Cross-region traffic management requires
        understanding which workloads are flexible, enforcing that residency-constrained
        workloads never enter the arbitrage pool, monitoring inter-region data transfer
        costs (which can eat into the savings), and maintaining the routing logic that
        evaluates current capacity pricing across all regions in near-real-time. This is
        tractable engineering at the hyperscaler level; it is a meaningful investment for
        smaller operators and generally only makes sense once the compute bill is large
        enough that a 30% reduction justifies a team maintaining the routing layer.
      </Prose>

      <Callout accent="gold">
        At global scale, region selection is a capacity-and-cost optimization as much as a
        latency one. The cheapest token is often the one served by a GPU in whichever
        region had spare supply this week.
      </Callout>

      <H3>The consistency question</H3>

      <Prose>
        Users are not stationary. A mobile user traveling from London to New York moves
        from EU-west to us-east mid-session. A developer using a VPN hits a different
        region than their physical location would suggest. A team's shared assistant session
        is accessed from offices in three cities. In each case, the user's conceptual
        session is continuous; the infrastructure's view of it is not. KV caches are
        region-local and don't follow the user. The new region starts cold — no cached
        context, no preserved state from the previous region's serving stack. For short
        conversations, this is barely noticeable. For long-running agent sessions or
        conversations with substantial context, the degradation is visible: the model has
        lost everything it "knew" from the first half of the conversation.
      </Prose>

      <Prose>
        Most providers accept this as a minor degradation and document it. A small number
        of latency-critical agent platforms have implemented cross-region session state
        transfer — serializing KV cache state and shipping it to the destination region
        on session migration — but the engineering cost is significant and the residency
        implications are non-trivial (you are now moving user-derived state across a region
        boundary, which is exactly what residency rules restrict). The honest current
        answer is that cross-region session continuity is an open problem in production
        systems, handled either by accepting the degradation or by keeping users pinned to
        a single region for the duration of a session, which reintroduces latency problems
        for traveling users. There is no clean solution that satisfies latency, residency,
        and continuity simultaneously.
      </Prose>

      <Prose>
        The AI Inference System Design section has walked the whole stack: architecture,
        routing, autoscaling, disaggregation, caching, multi-model serving, rate limits,
        safety, observability, streaming, cost, edge, and now global. The through-line is
        that inference is no longer a single-box problem. It is a distributed systems
        problem with LLM-specific hazards layered on top of every classical distributed-
        systems hazard — and those hazards compound. A KV cache that is fast to hit and
        correct to evict becomes a residency violation if it crosses the wrong region
        boundary. A routing algorithm that minimizes latency becomes a compliance failure
        if it ignores data-jurisdiction constraints. The classical infrastructure problems
        do not disappear when you add an LLM; they remain and acquire new consequences.
        The next section turns to a different dimension entirely — the long-context and
        retrieval layer that frequently drives the requests this whole infrastructure serves.
      </Prose>
    </div>
  ),
};

export default multiRegionGlobal;
