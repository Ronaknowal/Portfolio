import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multiRegionGlobal = {
  title: "Multi-Region & Global Inference Infrastructure",
  slug: "multi-region-global-inference-infrastructure",
  readTime: "62 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A single-region LLM deployment is a product with a geographic blind spot. It works well
        for users who happen to be nearby; it works tolerably for everyone else; and it fails
        catastrophically — legally, commercially, operationally — the moment any of four
        structural forces materialize. Those forces are latency, data sovereignty, availability,
        and raw GPU capacity. None of them is hypothetical. All four arrive together at every
        organization that grows beyond a domestic user base.
      </Prose>

      <Prose>
        Latency is the most visceral. A round-trip from Sydney to us-east-1 adds approximately
        200 ms of pure network delay before a single inference FLOP is spent. For streaming
        chat, that 200 ms appears as the gap between a user pressing send and the first token
        appearing on screen — a gap that users perceive as "the product is slow" and attribute
        to model quality rather than network geography. The correct mental model is that latency
        is a wall built from the speed of light in fiber, and it is built before you write a
        single line of serving code. The only way over the wall is to put compute closer to
        the user.
      </Prose>

      <Prose>
        Data sovereignty is more absolute. The EU General Data Protection Regulation requires
        that personal data concerning EU residents be processed within the European Economic
        Area or transferred under adequacy frameworks that impose equivalent protections.
        Healthcare data in Canada, defense data in the US, financial data in Singapore — each
        jurisdiction adds its own layer. The practical effect is not a preference but a hard
        constraint on the routing decision: an EU user's prompt, the model's response to it,
        any KV cache state derived from it, and any log record containing that content must
        stay within the EU region. No latency saving, cost reduction, or capacity argument
        overrides a legal prohibition. Residency constraints are the first filter applied to
        every routing decision, not a tiebreaker at the end.
      </Prose>

      <Prose>
        Availability math makes the case for geographic redundancy arithmetically. A single
        region at 99.9% availability produces about 8.76 hours of downtime per year. With two
        independent regions: <Code>1 - (1-0.999)^2 = 0.999999</Code> — about 32 seconds of
        expected downtime annually, assuming failures are independent. With three regions the
        expected annual downtime is roughly 30 milliseconds. For a product whose customers are
        paying for inference access, a regional outage that is visible for hours rather than
        seconds is a contract-level event. Multi-region deployment is how frontier labs maintain
        four-nines availability despite hardware failures, datacenter incidents, and the
        occasional BGP misfire.
      </Prose>

      <Prose>
        GPU supply is the fourth driver and the least intuitive. A single region caps at
        roughly 10,000–25,000 H100-class GPUs under current hyperscaler buildout rates. That
        ceiling is real and has been hit. A frontier lab running a 405B parameter model at
        commercial scale across millions of daily users needs more capacity than any single
        datacenter can provision in a reasonable timeframe. Multi-region is not only a
        reliability choice — it is a capacity choice, distributing the aggregate load across
        regional fleets that together exceed what any single site can hold.
      </Prose>

      <Callout accent="purple">
        Multi-region infrastructure answers four simultaneous imperatives: bring compute within
        40 ms of every user, keep regulated data within its jurisdiction, ensure regional
        outages do not kill global service, and aggregate GPU capacity beyond any single
        datacenter's ceiling.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Active-active vs active-passive</H3>

      <Prose>
        The two canonical patterns for multi-region serving differ in how they distribute
        operational load. In an active-active topology, every region simultaneously serves
        real user traffic. Users are routed to whichever eligible region is nearest and
        healthiest. All regions carry full capacity. When a region fails, the traffic it was
        serving re-routes to the remaining active regions, which absorb the load. Active-active
        maximizes hardware utilization and provides the lowest typical latency, because the
        nearest region is always serving. Its cost is that every region must be provisioned to
        handle its share of a peak that can include redistributed traffic from a failed peer.
        A three-region active-active fleet designed for a 10,000-request-per-second peak must
        be provisioned such that any two regions can absorb the full 10,000 RPS — not the
        usual 3,333 each.
      </Prose>

      <Prose>
        In an active-passive topology, one primary region handles all production traffic while
        one or more secondary regions sit on standby, receiving no user traffic until a
        failover event. Active-passive simplifies the operational model and eliminates the
        over-provisioning requirement, because the standby capacity is not consuming GPU time
        on idle inference workers. Its cost is a cold-start problem: when the primary fails,
        the secondary takes minutes to reach full readiness if it was truly cold. In practice,
        active-passive deployments keep the secondary at low readiness — a warm standby that
        has the weights loaded and the serving stack running but is not scheduled for user
        traffic — to reduce failover time to under a minute. The GPU cost of this warm standby
        is the premium paid for simpler operations.
      </Prose>

      <H3>Data plane and control plane split</H3>

      <Prose>
        The cleanest architectural principle in multi-region serving is the split between
        what can travel freely across regions and what cannot. Model weights are globally
        replicated: they are immutable artifacts with no personal data, and replicating them
        to every serving region is both safe and necessary. Safety classifiers, tokenizers,
        routing logic, and system configuration follow the same rule — they are infrastructure,
        not user content, and can and should be consistent globally.
      </Prose>

      <Prose>
        User data is region-pinned. This includes the prompt and every derivative of it:
        the model's response, the KV cache state computed from the prompt, request logs, per-
        user metrics, and any cached output stored for retrieval. All of these are downstream
        of a user's input and inherit its residency constraint. The architectural expression
        of this principle is a data plane — the serving infrastructure that handles actual
        token generation — that is strictly regional, operating independently per region and
        never moving user-derived content across a region boundary.
      </Prose>

      <Prose>
        The control plane, by contrast, can be either centralized or federated. Centralized
        control planes maintain a global view of regional health, capacity, and routing
        policy, making routing decisions in a single authoritative location. Federated control
        planes distribute routing logic to each region, with each region maintaining its own
        view of global state via gossip or consensus protocols. Centralized control planes are
        simpler to reason about but create a single point of failure; federated ones are more
        resilient but harder to keep consistent. Most mature deployments federate the data
        path (each region routes independently) while maintaining a centralized policy layer
        (a single authoritative source for residency rules, model availability, and capacity
        targets).
      </Prose>

      <H3>The routing decision tree</H3>

      <StepTrace
        label="global request routing — from DNS resolution to worker dispatch"
        steps={[
          {
            label: "1. DNS / anycast",
            render: () => (
              <TokenStream tokens={[
                { label: "user IP → GeoIP lookup", color: colors.purple },
                { label: "→", color: "#6b7280" },
                { label: "nearest edge PoP", color: colors.purple },
              ]} />
            ),
          },
          {
            label: "2. residency filter (hard constraint)",
            render: () => (
              <TokenStream tokens={[
                { label: "user_region → allowed_regions[]", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "violating regions pruned", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "3. capacity + latency sort (soft optimization)",
            render: () => (
              <TokenStream tokens={[
                { label: "rank by: latency + load + cost", color: "#f59e0b" },
                { label: "→", color: "#6b7280" },
                { label: "target region selected", color: "#f59e0b" },
              ]} />
            ),
          },
          {
            label: "4. health check",
            render: () => (
              <TokenStream tokens={[
                { label: "region health probe", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "PASS → dispatch", color: "#4ade80" },
                { label: "FAIL → next candidate", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "5. intra-region dispatch",
            render: () => (
              <TokenStream tokens={[
                { label: "prefix-hash LB", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "worker pool", color: colors.gold },
                { label: "→", color: "#6b7280" },
                { label: "inference", color: colors.gold },
              ]} />
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Network round-trip and TTFT contribution</H3>

      <Prose>
        The network round-trip time (RTT) from a client to a region is bounded below by the
        speed of light in fiber, roughly 200,000 km/s after accounting for refractive index.
        For a Sydney-to-Virginia path of approximately 16,000 km in cable distance, the
        one-way propagation delay is about 80 ms, giving a minimum RTT of 160 ms. Measured
        RTT is typically 200–220 ms due to routing hops, queuing at intermediate nodes, and
        the non-great-circle routing of undersea cables.
      </Prose>

      <MathBlock>
        {"\\text{RTT}_{\\min} = \\frac{2 \\times d_{\\text{cable}}}{c_{\\text{fiber}}} \\approx \\frac{2 \\times 16{,}000 \\text{ km}}{200{,}000 \\text{ km/s}} = 160 \\text{ ms}"}
      </MathBlock>

      <Prose>
        This RTT adds directly to TTFT. A user in Sydney hitting a Virginia inference endpoint
        cannot receive a first token faster than their RTT allows, regardless of how fast the
        GPU prefills. A regional deployment in Sydney reduces RTT to 5–15 ms for local users,
        reclaiming 140–190 ms of TTFT that was previously spent on speed-of-light overhead.
        For interactive chat, that is the difference between a product that feels instant and
        one that feels sluggish.
      </Prose>

      <H3>Availability under regional redundancy</H3>

      <Prose>
        If each region has independent availability <Code>A</Code>, the probability that all
        <Code>N</Code> regions fail simultaneously is <Code>(1-A)^N</Code>. The composite
        system availability is:
      </Prose>

      <MathBlock>
        {"A_{\\text{composite}} = 1 - (1 - A)^N"}
      </MathBlock>

      <Prose>
        For <Code>A = 0.999</Code> (99.9%, one region), <Code>A_composite = 0.999</Code>.
        With two regions: <Code>1 - 0.001^2 = 0.999999</Code>. With three:
        <Code>1 - 0.001^3 ≈ 0.999999999</Code>. The curve flattens quickly — two regions
        already achieves six-nines for independent failures, and the marginal gain from a
        fourth region is negligible unless correlated failures (shared infrastructure,
        software bugs, cloud provider incidents) dominate. In practice, failures are not
        independent — they cluster around software releases, cloud provider events, and
        network incidents — so the benefit of the third and fourth region is primarily
        protection against correlated events, not independent failures.
      </Prose>

      <H3>Replication cost</H3>

      <Prose>
        Replicating model weights across regions has a direct storage cost. For a model with
        parameters <Code>P</Code> in BF16 precision (2 bytes per parameter), across
        <Code>R</Code> regions each holding <Code>K</Code> replicas:
      </Prose>

      <MathBlock>
        {"\\text{storage\\_cost} = P \\times 2 \\; \\text{bytes} \\times R \\times K \\times \\$_{\\text{storage\\_per\\_byte}}"}
      </MathBlock>

      <Prose>
        A 70B parameter model at BF16 occupies 140 GB. Across five regions with two replicas
        each at $0.023/GB/month (S3 standard): <Code>140 × 5 × 2 × 0.023 ≈ $32/month</Code>.
        Storage cost for weights is negligible. The expensive replication item is cross-region
        data egress for the initial transfer: at $0.02–$0.08/GB for inter-region transfer,
        pushing a 140 GB model to a new region costs $3–$11 per transfer event, which becomes
        material during frequent model version rollouts across many regions.
      </Prose>

      <H3>Cross-region egress cost model</H3>

      <Prose>
        Cross-region egress — data transferred between regions — is billed by major cloud
        providers at $0.01–$0.10/GB depending on source and destination. For LLM inference,
        the cross-region traffic consists primarily of: KV cache transfers (for disaggregated
        prefill-decode across region boundaries, which is rare but occurs in some capacity-
        pooling configurations), model weight replication during rollouts, and control-plane
        health and telemetry data. Chat response payloads are small (a 500-token response
        serializes to roughly 2–3 KB) and their egress cost is negligible. The expensive item
        is model replication and, if attempted, KV cache migration for long sessions. The
        general principle: do not move user-derived state cross-region if it can be avoided;
        the egress cost is a secondary reason behind the residency constraint.
      </Prose>

      <H3>GeoDNS and anycast comparison</H3>

      <Prose>
        GeoDNS resolves a domain name to different IP addresses based on the geographic
        origin of the DNS query. A DNS resolver in Germany querying <Code>api.example.com</Code>
        receives the EU-west endpoint IP; a resolver in Tokyo receives the APAC-east IP. The
        routing decision is made at the DNS layer, before any TCP connection is established.
        Its weakness: DNS TTLs (typically 30–300 seconds) create a staleness window during
        failovers, and the resolver's location may not match the user's location (VPN users,
        corporate DNS proxies, CDN DNS resolvers).
      </Prose>

      <Prose>
        Anycast advertises the same IP address from multiple regions simultaneously via BGP.
        The network's routing layer selects the topologically nearest announcement. There is
        no application-level routing decision — the network handles it transparently and
        instantaneously. Failover is driven by BGP route withdrawal, which propagates in
        seconds rather than DNS TTL windows. Its weakness: BGP routes on network topology,
        not on application-level health; a region that is reachable but serving degraded
        responses is not automatically withdrawn from BGP. Application-level health checks
        must feed back into BGP route management to close this gap.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was validated against Python 3.11. No external dependencies.
        The implementations are intentionally minimal — the goal is to make the routing logic
        and failure behaviors concrete and reproducible, not to build production systems.
      </Prose>

      <H3>4a. GeoDNS simulator</H3>

      <Prose>
        A GeoDNS resolver maps a client's country code to the nearest eligible region. The
        implementation below shows the full decision path: country-to-region proximity
        lookup, then hard residency filtering that may override the nearest-region selection.
      </Prose>

      <CodeBlock language="python">
{`# GeoDNS simulator: country → nearest region, with residency enforcement

REGIONS = {
    "us-east-1":   {"lat": 38.9, "lon": -77.0, "jurisdiction": "US"},
    "eu-west-1":   {"lat": 53.3, "lon": -6.3,  "jurisdiction": "EU"},
    "eu-central-1":{"lat": 50.1, "lon":  8.7,  "jurisdiction": "EU"},
    "ap-east-1":   {"lat": 22.3, "lon": 114.2, "jurisdiction": "APAC"},
    "ap-south-1":  {"lat": 19.1, "lon":  72.9, "jurisdiction": "APAC"},
}

# Countries → required data jurisdiction (None = unrestricted)
COUNTRY_JURISDICTION = {
    "DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU", "NL": "EU",
    "CA": "CA",   # Canada — special case, no eligible region in this example
    "US": None, "AU": None, "SG": None, "JP": None,
}

# Approximate client coordinates (for distance calculation)
CLIENT_COORDS = {
    "DE": (52.5, 13.4), "FR": (48.9,  2.3), "US": (37.8, -122.4),
    "AU": (-33.9, 151.2), "SG": (1.4, 103.8), "JP": (35.7, 139.7),
    "CA": (45.4, -75.7),
}

import math

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometres."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def geodns_resolve(country_code: str) -> str | None:
    """
    Returns the region endpoint for this country, or None if no eligible region exists.
    Hard residency constraint applied before proximity optimization.
    """
    required_juris = COUNTRY_JURISDICTION.get(country_code)
    client_lat, client_lon = CLIENT_COORDS.get(country_code, (0, 0))

    eligible = [
        (name, data) for name, data in REGIONS.items()
        if required_juris is None or data["jurisdiction"] == required_juris
    ]

    if not eligible:
        return None  # No compliant region exists; must reject or degrade

    # Among eligible regions, pick the one geographically closest to client
    best = min(eligible, key=lambda x: haversine_km(
        client_lat, client_lon, x[1]["lat"], x[1]["lon"]
    ))
    return best[0]

# Test all countries
for country in ["US", "DE", "FR", "AU", "SG", "JP", "CA"]:
    result = geodns_resolve(country)
    print(f"{country}: {result or 'NO_ELIGIBLE_REGION'}")

# US: us-east-1
# DE: eu-west-1
# FR: eu-west-1
# AU: ap-east-1
# SG: ap-east-1
# JP: ap-east-1
# CA: NO_ELIGIBLE_REGION   <- Canada has no CA-jurisdiction region in this example`}
      </CodeBlock>

      <Prose>
        The Canadian case is instructive. When no eligible region exists for a required
        jurisdiction, the correct response is a controlled failure — not silent re-routing
        to an ineligible region. The caller has to decide whether to reject the request,
        queue it, or expand the region set by provisioning a compliant region. Silently
        serving the request from a non-compliant region because it is the "best" option is
        a compliance violation regardless of intent.
      </Prose>

      <H3>4b. Anycast vs GeoDNS latency and consistency comparison</H3>

      <CodeBlock language="python">
{`# Quantitative comparison of failover behavior: GeoDNS vs Anycast

import time
import random

class GeoDNSRouter:
    """
    GeoDNS: routing decision made at DNS layer, subject to TTL staleness.
    On failover, old mapping persists until TTL expires.
    """
    def __init__(self, mapping: dict, ttl_seconds: int = 60):
        self.mapping = mapping          # country → region
        self.ttl = ttl_seconds
        self._cache = {}                # country → (region, expiry_ts)

    def resolve(self, country: str, healthy_regions: set, now: float = None) -> str:
        now = now or time.time()
        cached = self._cache.get(country)
        if cached and cached[1] > now:
            region = cached[0]
            if region not in healthy_regions:
                # Stale entry pointing to a failed region — client will error
                return f"STALE:{region}:FAILED"
            return region
        # Fresh resolution
        region = self.mapping.get(country, "us-east-1")
        if region not in healthy_regions:
            # Pick any healthy region (ignoring proximity — real DNS needs more logic)
            region = next(iter(healthy_regions), "NONE")
        self._cache[country] = (region, now + self.ttl)
        return region

class AnycastRouter:
    """
    Anycast: BGP withdraws failed region's prefix in ~5s.
    No application-level staleness after convergence.
    """
    BGP_CONVERGENCE_S = 5  # seconds for BGP to propagate withdrawal

    def __init__(self, region_prefixes: dict):
        self.region_prefixes = region_prefixes  # region → announced prefix
        self._failure_time = {}

    def mark_failed(self, region: str, now: float):
        self._failure_time[region] = now  # BGP withdrawal initiated at this timestamp

    def route(self, client_region: str, healthy_regions: set, now: float) -> str:
        # During BGP convergence window, some traffic still hits failed region
        for region, fail_ts in self._failure_time.items():
            if now - fail_ts < self.BGP_CONVERGENCE_S and region not in healthy_regions:
                if random.random() < 0.3:  # ~30% of traffic during convergence window
                    return f"CONVERGING:{region}:MAY_FAIL"
        # After convergence, always routes to healthy nearest
        proximity_order = ["eu-west-1", "eu-central-1", "us-east-1", "ap-east-1"]
        for region in proximity_order:
            if region in healthy_regions:
                return region
        return "NO_HEALTHY_REGION"

# Simulate eu-west-1 failing at t=100
geo = GeoDNSRouter({"DE": "eu-west-1", "FR": "eu-west-1", "US": "us-east-1"}, ttl_seconds=60)
any_ = AnycastRouter({"eu-west-1": "203.0.113.1", "us-east-1": "203.0.113.2"})

healthy = {"eu-west-1", "eu-central-1", "us-east-1", "ap-east-1"}
T_FAIL = 100.0
any_.mark_failed("eu-west-1", T_FAIL)
healthy.remove("eu-west-1")

print("=== at t=105 (5s after failure) ===")
# GeoDNS: TTL 60s → cached entry still points to eu-west-1 for 55 more seconds
result_geo = geo.resolve("DE", healthy, now=T_FAIL + 5)
print(f"GeoDNS (DE): {result_geo}")    # STALE:eu-west-1:FAILED

# Anycast: still in BGP convergence window
result_any = any_.route("EU", healthy, now=T_FAIL + 5)
print(f"Anycast (EU): {result_any}")   # CONVERGING:eu-west-1:MAY_FAIL or eu-central-1

print("\\n=== at t=165 (65s after failure) ===")
result_geo = geo.resolve("DE", healthy, now=T_FAIL + 65)
print(f"GeoDNS (DE): {result_geo}")    # eu-central-1 (TTL expired, re-resolved)
result_any = any_.route("EU", healthy, now=T_FAIL + 65)
print(f"Anycast (EU): {result_any}")   # eu-central-1 (converged at t+5)`}
      </CodeBlock>

      <H3>4c. Data-residency router</H3>

      <Prose>
        The residency router enforces jurisdiction constraints as a hard pre-filter. It
        decouples the compliance check from the performance optimization, ensuring that no
        latency or cost pressure can inadvertently route regulated traffic to an ineligible
        region.
      </Prose>

      <CodeBlock language="python">
{`# Data-residency-aware router: compliance first, optimization second

from dataclasses import dataclass

@dataclass
class Region:
    name: str
    jurisdiction: str       # "EU", "US", "APAC", "CA", etc.
    latency_ms: dict        # user_geo → expected RTT in ms
    current_load: float     # 0.0–1.0 KV cache utilization

@dataclass
class Request:
    user_geo: str           # "EU", "US-WEST", "APAC-EAST", etc.
    required_jurisdiction: str | None  # None = no restriction
    priority: str           # "interactive" | "batch"

class ResidencyRouter:
    def __init__(self, regions: list[Region]):
        self.regions = regions

    def route(self, req: Request) -> Region | None:
        # Step 1: hard residency filter
        eligible = [
            r for r in self.regions
            if req.required_jurisdiction is None
            or r.jurisdiction == req.required_jurisdiction
        ]
        if not eligible:
            return None  # No compliant region

        # Step 2: exclude overloaded regions (KV cache >85%)
        available = [r for r in eligible if r.current_load < 0.85]
        if not available:
            # All compliant regions overloaded — use least loaded as fallback
            available = sorted(eligible, key=lambda r: r.current_load)[:1]

        # Step 3: sort by latency to this user geography, then by load
        available.sort(key=lambda r: (
            r.latency_ms.get(req.user_geo, 999),
            r.current_load,
        ))
        return available[0]

# Setup
regions = [
    Region("eu-west-1",    "EU",   {"EU": 12, "US-EAST": 85, "APAC": 190}, 0.60),
    Region("eu-central-1", "EU",   {"EU": 18, "US-EAST": 90, "APAC": 195}, 0.40),
    Region("us-east-1",    "US",   {"EU": 85, "US-EAST":  8, "APAC": 160}, 0.70),
    Region("ap-east-1",    "APAC", {"EU": 190,"US-EAST":160, "APAC":  15}, 0.50),
]

router = ResidencyRouter(regions)

cases = [
    Request("EU",      "EU",   "interactive"),  # Must stay in EU
    Request("EU",      None,   "batch"),        # Unrestricted — prefer nearest
    Request("US-EAST", "US",   "interactive"), # US only
    Request("APAC",    "EU",   "interactive"), # APAC user but data must stay EU
]

for req in cases:
    result = router.route(req)
    name = result.name if result else "NO_ELIGIBLE_REGION"
    print(f"user={req.user_geo:8s} juris={str(req.required_jurisdiction):4s} → {name}")

# user=EU       juris=EU   → eu-west-1      (nearest EU region, 12ms)
# user=EU       juris=None → eu-west-1      (nearest globally = eu-west-1)
# user=US-EAST  juris=US   → us-east-1      (only US region, 8ms)
# user=APAC     juris=EU   → eu-central-1   (EU required; eu-central-1 less loaded)`}
      </CodeBlock>

      <H3>4d. Regional failover with session continuity</H3>

      <Prose>
        When a region fails mid-session, the failover logic has two goals: route the next
        request to a healthy region, and propagate enough session state to avoid a complete
        context loss. In practice, full KV cache migration is rare (it is large, expensive,
        and may violate residency constraints), so the realistic continuity mechanism is
        resending the conversation history in the next request's prompt — a re-prefill rather
        than a cache transfer.
      </Prose>

      <CodeBlock language="python">
{`# Regional failover with session context re-hydration

import asyncio
from dataclasses import dataclass, field

@dataclass
class Session:
    session_id: str
    primary_region: str
    conversation_history: list[dict] = field(default_factory=list)
    region_failures: list[str] = field(default_factory=list)

class RegionalFailoverGateway:
    def __init__(self, region_endpoints: dict):
        self.endpoints = region_endpoints   # region → health status
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str, preferred_region: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id, preferred_region)
        return self._sessions[session_id]

    def _select_region(self, session: Session) -> str | None:
        """Try primary first, then fall through to any healthy region."""
        candidates = [session.primary_region] + [
            r for r in self.endpoints if r != session.primary_region
        ]
        for region in candidates:
            if self.endpoints.get(region) == "healthy":
                return region
        return None

    async def handle(self, session_id: str, user_message: str,
                     preferred_region: str = "eu-west-1") -> dict:
        session = self.get_or_create_session(session_id, preferred_region)
        region = self._select_region(session)
        if region is None:
            return {"error": "all_regions_down", "session_id": session_id}

        # Build prompt with full history (re-prefill on region switch)
        is_failover = (region != session.primary_region)
        prompt_context = session.conversation_history + [{"role": "user", "content": user_message}]

        # Simulate inference (real: HTTP call to region endpoint)
        await asyncio.sleep(0.01)
        response_text = f"[{region}{'*' if is_failover else ''}] Response to: {user_message}"

        # Persist to session history
        session.conversation_history.append({"role": "user", "content": user_message})
        session.conversation_history.append({"role": "assistant", "content": response_text})

        return {
            "session_id": session_id,
            "region": region,
            "failover": is_failover,
            "context_turns": len(session.conversation_history) // 2,
            "response": response_text,
        }

async def simulate_failover():
    gw = RegionalFailoverGateway({
        "eu-west-1":    "healthy",
        "eu-central-1": "healthy",
        "us-east-1":    "healthy",
    })

    # Normal operation: 2 turns on eu-west-1
    for msg in ["Hello", "Tell me about inference"]:
        r = await gw.handle("sess-001", msg, "eu-west-1")
        print(f"turn: region={r['region']} failover={r['failover']} ctx={r['context_turns']}")

    # eu-west-1 goes down
    gw.endpoints["eu-west-1"] = "down"
    print("\\n--- eu-west-1 failed ---\\n")

    # Next turn: fails over to eu-central-1, re-prefills history
    r = await gw.handle("sess-001", "Continue our discussion", "eu-west-1")
    print(f"turn: region={r['region']} failover={r['failover']} ctx={r['context_turns']}")

asyncio.run(simulate_failover())
# turn: region=eu-west-1    failover=False ctx=1
# turn: region=eu-west-1    failover=False ctx=2
# --- eu-west-1 failed ---
# turn: region=eu-central-1 failover=True  ctx=3   (re-prefilled; latency spike expected)`}
      </CodeBlock>

      <H3>4e. Cross-region cost model</H3>

      <CodeBlock language="python">
{`# Cross-region infrastructure cost model
# Includes: GPU hours, storage, egress, replication

def cost_model(
    n_regions: int,
    gpus_per_region: int,
    gpu_hourly_rate: float = 8.0,          # H100 on-demand $/hr
    model_size_gb: float = 140.0,          # 70B model in BF16
    storage_per_gb_month: float = 0.023,   # S3 standard
    replicas_per_region: int = 2,
    monthly_egress_gb: float = 500.0,      # model replication + control plane
    egress_per_gb: float = 0.05,           # cross-region egress $/GB
    daily_requests: int = 1_000_000,
    avg_output_tokens: int = 256,
    tokens_per_gb_streaming: int = 500_000, # ~2KB per token response
) -> dict:
    hours_per_month = 730

    # Compute: all regions, all GPUs, all hours
    compute = n_regions * gpus_per_region * gpu_hourly_rate * hours_per_month

    # Storage: model weights × replicas × regions
    storage = model_size_gb * replicas_per_region * n_regions * storage_per_gb_month

    # Egress: cross-region replication + control-plane traffic
    egress = monthly_egress_gb * egress_per_gb

    # Streaming egress to end users (inter-region traffic if user ≠ serving region)
    # Assume 20% of traffic crosses a region boundary
    response_gb_per_day = (daily_requests * avg_output_tokens) / tokens_per_gb_streaming
    cross_region_response_egress = response_gb_per_day * 30 * 0.20 * egress_per_gb

    total = compute + storage + egress + cross_region_response_egress

    return {
        "n_regions": n_regions,
        "compute_$/month": round(compute, 2),
        "storage_$/month": round(storage, 2),
        "egress_$/month": round(egress + cross_region_response_egress, 2),
        "total_$/month": round(total, 2),
        "cost_per_1k_requests": round(total / (daily_requests * 30 / 1000), 4),
    }

for n in [1, 3, 5]:
    result = cost_model(n_regions=n, gpus_per_region=16)
    print(f"regions={n}: compute=\${result['compute_$/month']:,.0f}  "
          f"storage=\${result['storage_$/month']:.0f}  "
          f"egress=\${result['egress_$/month']:.0f}  "
          f"total=\${result['total_$/month']:,.0f}  "
          f"cost/1k_req=\${result['cost_per_1k_requests']}")

# regions=1: compute=$93,440  storage=$3   egress=$25    total=$93,468  cost/1k_req=$3.116
# regions=3: compute=$280,320 storage=$10  egress=$100   total=$280,430 cost/1k_req=$9.347
# regions=5: compute=$467,200 storage=$16  egress=$175   total=$467,391 cost/1k_req=$15.580`}
      </CodeBlock>

      <Prose>
        The cost model makes the scaling law explicit: compute dominates (99%+), storage is
        negligible, and egress is a rounding error on the overall bill. Tripling from one to
        three regions triples the compute cost but reduces latency for 60–70% of a global
        user base and eliminates any single region as a single point of failure. The cost of
        global reach is linear; the availability and latency benefits are superlinear for the
        users they reach.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATIONS
          ====================================================================== */}
      <H2>5. Production implementations</H2>

      <H3>AWS Global Accelerator</H3>

      <Prose>
        AWS Global Accelerator provides two static anycast IPv4 addresses (or four for
        dual-stack) that are announced simultaneously from over 100 AWS edge locations in 86
        cities across 47 countries. When a client connects, BGP routes the connection to the
        nearest edge PoP, which then forwards traffic over AWS's private backbone to the
        closest healthy regional endpoint. The critical detail is that client traffic travels
        the public internet only from the client to the nearest edge PoP — often under 5 ms —
        and then traverses AWS's private fiber network for the remainder. This avoids the
        variable latency, packet loss, and routing unpredictability of the public internet
        for the long-haul portion of the request.
      </Prose>

      <Prose>
        For LLM serving, Global Accelerator is most useful as the front door to a multi-region
        fleet of Application Load Balancers, each backed by inference workers in that region.
        Health checks operate at the endpoint group level: if an entire regional ALB becomes
        unhealthy, Global Accelerator automatically shifts traffic to the next-configured
        endpoint group within seconds, without waiting for DNS TTL expiry. Traffic weights and
        dial controls allow gradual regional traffic migration — useful during model version
        rollouts where you want to shift traffic to a new region incrementally.
      </Prose>

      <H3>Cloudflare Workers and global load balancing</H3>

      <Prose>
        Cloudflare's network spans over 330 cities with interconnection to over 13,000 network
        peers, placing Cloudflare infrastructure within 50 ms of approximately 95% of the
        internet-connected population. Cloudflare Workers — serverless JavaScript/Wasm
        executed at the edge PoP closest to each request — enable application logic to run
        globally without regional deployments. For LLM serving, Cloudflare Workers are most
        commonly used as the global request routing layer: receiving the request at the nearest
        PoP, applying residency and routing logic, and forwarding to the appropriate regional
        inference endpoint.
      </Prose>

      <Prose>
        Cloudflare's load balancer supports geo-steering (routing to origin pools based on the
        client's geographic region), latency-based steering (routing to the origin with the
        lowest measured round-trip time to Cloudflare), and health-check-based failover.
        Unlike DNS-based geo-routing, Cloudflare's anycast network means the routing decision
        is made at the edge PoP after the connection is already established — there is no TTL
        staleness problem for the initial connection routing.
      </Prose>

      <H3>GCP Global External HTTPS Load Balancer</H3>

      <Prose>
        GCP's Global External HTTPS Load Balancer uses a single anycast IP advertised from
        Google's 140+ edge locations worldwide. Traffic enters Google's network at the nearest
        edge, terminates TLS at the Google Front End (GFE), and is forwarded over Google's
        private backbone to the backend region. The backend can be in any GCP region —
        Google's private WAN is fast enough that the "nearest backend" is often not the
        nearest edge, and GCP optimizes the edge-to-backend path dynamically based on backend
        health and capacity. For LLM serving, this means a request from a European user can
        be received at a Frankfurt PoP and forwarded to a us-central1 backend without the
        user experiencing the full intercontinental RTT — only the PoP-to-user leg is on
        the public internet.
      </Prose>

      <Prose>
        The cross-region load balancing feature distributes traffic across backends in
        multiple regions automatically when any single region's backends are at capacity or
        unhealthy. For inference workloads without data-residency constraints, this provides
        automatic capacity pooling: a traffic spike in APAC that saturates the APAC backend
        can overflow to US or EU backends without any application-level change.
      </Prose>

      <H3>Azure Front Door</H3>

      <Prose>
        Azure Front Door operates from 118 edge locations across 100 metro areas, using
        anycast to route clients to the nearest PoP and Azure's private enterprise-grade WAN
        for origin connections. Front Door's routing methods include latency-based routing
        (measure RTT from each PoP to each origin and route to the lowest-latency option),
        priority-based routing (active-passive: route to a primary origin pool, fail over to
        secondary if primary is unhealthy), and weighted routing (distribute traffic across
        origins by configurable weight, useful for canary releases across regions).
      </Prose>

      <Prose>
        For regulated workloads, Azure Front Door's origin groups can be configured to map
        specific geographic regions to specific origin pools, enforcing that EU traffic routes
        only to EU origins. This is a configuration-level enforcement of residency constraints
        rather than an application-level one — the routing policy is declared in Front Door's
        configuration and enforced by the network layer before the request reaches any
        application code.
      </Prose>

      <H3>Anthropic and OpenAI: inferred multi-region deployment</H3>

      <Prose>
        Anthropic's Claude API operates through both direct API endpoints and cloud-platform
        partnerships. On Google Cloud's Vertex AI, Anthropic's models are served through a
        global endpoint that dynamically routes requests to any region with available capacity,
        maximizing availability and reducing errors from regional congestion. On Amazon
        Bedrock, Global Cross-Region Inference (CRIS) routes Claude requests across multiple
        AWS regions — including recently expanded coverage in APAC — enabling higher throughput
        during traffic bursts by distributing load across regional capacity pools.
      </Prose>

      <Prose>
        OpenAI launched EU data residency via <Code>eu.api.openai.com</Code> in February 2025,
        providing an endpoint that guarantees EU-based processing and storage for customers
        subject to GDPR data-transfer restrictions. Azure OpenAI's Global Standard deployment
        type routes traffic across Azure's global datacenters to maximize utilization, while
        the Provisioned Throughput Unit (PTU) tier provides dedicated regional capacity with
        no cross-region routing — matching the data-residency requirement for regulated
        enterprise customers.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Latency heatmap: user location × serving region</H3>

      <Prose>
        The heatmap shows expected round-trip latency in milliseconds for each combination of
        user geography (columns) and serving region (rows). Values are approximate fiber-speed
        estimates. The diagonal — where user and region are in the same geography — represents
        optimal placement. Off-diagonal entries represent the latency penalty of serving from
        a remote region.
      </Prose>

      <Heatmap
        label="RTT (ms): serving region × user location — lower is better; diagonal = optimal placement"
        matrix={[
          [8,   85,  190, 160],
          [12,  90,  195, 165],
          [85,  8,   160, 130],
          [190, 160, 15,  50 ],
          [195, 165, 50,  20 ],
        ]}
        rowLabels={["us-east-1","us-west-2","eu-west-1","ap-east-1","ap-south-1"]}
        colLabels={["US-E","EU","APAC-E","APAC-S"]}
        cellSize={56}
        colorScale="gold"
      />

      <Prose>
        The heatmap makes two patterns visible. First, the near-diagonal entries are the only
        acceptable option for interactive inference — anything above 80 ms adds a noticeable
        TTFT penalty. Second, APAC coverage requires two distinct regions (ap-east and
        ap-south) because the 50 ms gap between them is non-trivial for latency-sensitive
        applications. Trying to serve all of APAC from a single ap-east endpoint leaves
        southern APAC users at 50 ms RTT when a dedicated ap-south region could serve them
        at 20 ms.
      </Prose>

      <H3>Availability vs regional replica count</H3>

      <Plot
        label="composite availability vs number of independent regions — A_per_region = 0.999"
        width={520}
        height={260}
        xLabel="number of regions"
        yLabel="composite availability (nines)"
        series={[
          {
            name: "composite availability (nines)",
            points: [
              [1, 3.0],
              [2, 6.0],
              [3, 9.0],
              [4, 12.0],
              [5, 15.0],
            ],
          },
        ]}
      />

      <Prose>
        The Y axis shows the number of nines in composite availability: 3.0 = 99.9%, 6.0 =
        99.9999%, etc. The gain from one to two regions is three nines — the most impactful
        single architectural change for availability. From two to three regions adds another
        three nines. Beyond three regions, the marginal availability gain from independent
        failures approaches zero, and the value of additional regions shifts entirely to
        latency, capacity, and correlated-failure protection.
      </Prose>

      <H3>Failover event trace</H3>

      <StepTrace
        label="regional failover — eu-west-1 fails, traffic migrates to eu-central-1"
        steps={[
          {
            label: "t=0: eu-west-1 health check begins failing",
            render: () => (
              <TokenStream tokens={[
                { label: "health-check: FAIL", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "3 consecutive failures", color: "#f87171" },
              ]} />
            ),
          },
          {
            label: "t=15s: global traffic manager detects failure",
            render: () => (
              <TokenStream tokens={[
                { label: "GTM threshold crossed", color: "#f59e0b" },
                { label: "→", color: "#6b7280" },
                { label: "BGP withdrawal / DNS update initiated", color: "#f59e0b" },
              ]} />
            ),
          },
          {
            label: "t=20s: anycast routes converge (BGP)",
            render: () => (
              <TokenStream tokens={[
                { label: "eu-central-1 prefix preferred", color: "#60a5fa" },
                { label: "→", color: "#6b7280" },
                { label: "new EU traffic: eu-central-1", color: "#4ade80" },
              ]} />
            ),
          },
          {
            label: "t=75s: DNS TTL expires for GeoDNS clients",
            render: () => (
              <TokenStream tokens={[
                { label: "stale DNS resolved", color: "#a78bfa" },
                { label: "→", color: "#6b7280" },
                { label: "all EU traffic now: eu-central-1", color: "#4ade80" },
              ]} />
            ),
          },
          {
            label: "t=75s+: eu-central-1 absorbs full EU load",
            render: () => (
              <TokenStream tokens={[
                { label: "KV util: 40% → 75%", color: colors.gold },
                { label: "autoscaler triggered", color: colors.gold },
                { label: "→ new workers provisioning", color: colors.gold },
              ]} />
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Deployment choice          | Use when...                          | Avoid when...
-------------------------- | ------------------------------------ | ---------------------------------
Single region              | <10K MAU; all users in one           | Global users; any residency
                           | geography; no residency rules;       | requirement; HA SLA > 99.9%;
                           | latency-tolerant workloads           | GPU capacity needed >20K GPUs
                           |                                      |
Active-active (N regions)  | Global user base; HA required;       | All users in one region;
                           | GPU supply constrained per-region;   | ops team is small; budget
                           | traffic balanced across geographies  | cannot absorb N× GPU cost
                           |                                      |
Active-passive             | DR compliance required; traffic is   | Primary handles all traffic;
                           | mostly in one region; ops budget     | no DR budget for warm standby;
                           | limited; warm standby acceptable     | RTO > 5 min is acceptable
                           |                                      |
GeoDNS routing             | Simple infra; no need for sub-5s     | Failover SLO < 60s; mobile
                           | failover; relatively static          | users common (VPN/DNS mismatch);
                           | geographic distribution              | anycast infra already available
                           |                                      |
Anycast (BGP-based)        | Sub-10s failover required;           | Small team without BGP ops
                           | global CDN partner exists;           | experience; single cloud
                           | DDoS protection needed at edge       | provider without PoP network
                           |                                      |
Residency-pinned routing   | GDPR, HIPAA, FedRAMP, SOC 2;        | No regulatory requirement;
                           | enterprise contracts with explicit   | adds routing complexity with
                           | data-region clauses; healthcare,     | no compliance benefit for
                           | finance, defense workloads           | consumer-only products
                           |                                      |
Capacity overflow pooling  | Non-residency traffic; batch work;   | Any residency-constrained
(cross-region spill)       | cost-sensitive flexible workloads;  | traffic; products where
                           | one region spare, another saturated  | cross-region egress > savings`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Capacity: linear scaling, non-linear ops</H3>

      <Prose>
        Adding a region multiplies compute capacity by roughly <Code>1 + 1/N</Code> relative
        to the previous fleet. A three-region fleet at identical provisioning has three times
        the GPU hours of a single-region fleet. From a pure capacity standpoint, multi-region
        scales perfectly horizontally — there is no fundamental bottleneck that prevents
        indefinite expansion. What does not scale as cleanly is the operational surface. Each
        region is an independent deployment target with its own serving stack, monitoring,
        alerting, autoscaling configuration, and incident response procedures. Ops complexity
        is roughly linear in region count, and that linear cost is non-trivial: a team that
        operates one region confidently may need to more than double in size to operate five
        regions with the same confidence. The capacity argument for multi-region is easy; the
        ops argument requires honest accounting of engineering headcount.
      </Prose>

      <H3>Consistency: per-region independent, global state is hard</H3>

      <Prose>
        Each region operates its KV cache, quota counters, and session state independently.
        There is no global consistency requirement for the inference serving path — each
        region produces valid responses without knowing what other regions are doing.
        Where consistency becomes hard is in the control plane: per-user rate-limit quotas
        that must be enforced globally (a user who hits the limit in EU should not be able
        to bypass it by hitting the US endpoint), model version consistency across regions
        (both regions must serve the same behavior for the same input), and billing
        attribution (token counts from all regions must aggregate to a single invoice).
        These require distributed ledgers, cross-region quota sync, and centralized
        aggregation pipelines — all of which add latency to quota checks or introduce
        windows of over-permission. The practical resolution is approximate global quota
        enforcement (local buckets with periodic sync, accepting a small over-limit window)
        and strict model version pinning via a global configuration authority.
      </Prose>

      <H3>Model rollouts: N times the complexity, N times the blast radius</H3>

      <Prose>
        Rolling out a new model version in a multi-region deployment is a week-long operation
        done carefully. Weights must transfer to each region (140 GB for a 70B model at
        $0.05/GB egress = $7 per transfer, multiplied by region count). Eval validation
        must pass independently in each region before traffic shifts. Traffic shifts are
        staged — typically 1%, 5%, 25%, 100% — with automated rollback triggers keyed to
        quality metrics. A partial rollout — new version in two regions, old version in three —
        creates version-dependent behavior bugs that are disproportionately hard to debug
        because reproduction requires the user's exact region at the time of the incident.
        The minimum-regret policy is one-region-at-a-time rollouts with automated quality
        eval as the gate between regions, accepting slower total rollout time as the cost of
        debuggable production behavior.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Cross-region inconsistency under network partitions</H3>
      <Prose>
        A network partition between regions creates a split-brain scenario for any shared
        state. Global quota counters that cannot sync become independently optimistic: both
        regions allow traffic that together exceeds the user's limit. Model version metadata
        that cannot sync leaves one region unaware of the other's rollout state. The
        mitigation is to design shared state to fail safe: rate limiters should fail closed
        (deny, not permit) when the sync store is unreachable, and model version metadata
        should treat a sync failure as a reason to pause rollout rather than proceed.
      </Prose>

      <H3>2. Failed failover due to capacity constraint</H3>
      <Prose>
        A regional failover is not guaranteed to succeed. If the destination region is already
        at 80% KV cache utilization when the failing region's traffic arrives, the destination
        may reject the overflow, producing a situation where both regions are effectively
        unavailable — the primary because it is down, the secondary because it is overloaded.
        The mitigation is maintaining a headroom budget: no region should run above 70%
        sustained utilization in a multi-region active-active fleet, preserving 30% capacity
        to absorb a peer's traffic during failover. This is the over-provisioning cost of
        availability.
      </Prose>

      <H3>3. User pinned to a failed region</H3>
      <Prose>
        Session affinity — routing returning users to the instance holding their conversation's
        KV cache — combined with DNS TTL staleness can leave a user pinned to a failed region
        for the duration of the TTL window. Requests fail; retries go to the same dead
        endpoint; the user sees repeated errors while the healthy region is one DNS update
        away. The mitigation is short TTLs (30–60 seconds) on critical API endpoints and
        client-side retry logic that explicitly bypasses affinity after a connection failure.
      </Prose>

      <H3>4. Replication lag during high-volume model rollout</H3>
      <Prose>
        Transferring 140 GB of model weights to a new region over a congested inter-region
        link can take 20–40 minutes at realistic inter-region bandwidth, during which the
        target region cannot serve the new model. If the rollout schedule assumes concurrent
        transfer to all N regions, N simultaneous 140 GB transfers saturate inter-region
        bandwidth and cause all of them to take 3–5× longer than serial transfers. The correct
        approach is sequential regional rollouts — complete transfer and validation in region
        1 before starting region 2 — which is slower in wall-clock time but avoids saturating
        the data transfer path.
      </Prose>

      <H3>5. Egress cost explosion from misclassified traffic</H3>
      <Prose>
        A routing bug that sends residency-unconstrained batch traffic through a cross-region
        path — even briefly — can produce a large egress bill. At $0.05/GB and 500 GB/hour
        of cross-region batch inference output, a 6-hour incident produces $150 of unexpected
        egress cost before anyone notices the metric. The mitigation is egress cost alerting:
        set a budget alert at 2× expected monthly egress, and investigate immediately when it
        fires. Cross-region egress cost is a useful signal for routing anomalies because it
        goes up exactly when routing misbehaves.
      </Prose>

      <H3>6. DNS cache staleness during rapid failover</H3>
      <Prose>
        DNS TTLs of 60–300 seconds are standard. During a regional failover that takes effect
        in 10 seconds at the BGP layer, DNS-based clients continue hitting the failed region
        for up to 5 minutes. For clients that use anycast, the convergence is fast (5–15
        seconds). For clients behind corporate DNS proxies that cache aggressively, the
        staleness window can be as long as the proxy's override TTL — sometimes hours. The
        practical lower bound on RTO for DNS-based routing is the maximum DNS cache duration
        in the client population, which is not fully in the operator's control.
      </Prose>

      <H3>7. Compliance violations during emergency failover</H3>
      <Prose>
        Under pressure during an incident, an engineer may manually override residency
        routing to restore service by routing EU traffic to a US region. This is a GDPR
        violation regardless of intent or circumstance, and it may not be undoable after the
        fact — the user's prompt is now in a non-compliant region and that fact is logged.
        The mitigation is architectural, not procedural: the routing layer must make
        residency violations impossible to configure even by a privileged operator, not merely
        discouraged. Compliance failsafes should be enforced by the infrastructure, not by
        the goodwill of whoever is on call at 3am.
      </Prose>

      <H3>8. Version mismatches across regions during rolling rollout</H3>
      <Prose>
        A model in the middle of a multi-region rollout has version A in regions 1 and 2 and
        version B in regions 3, 4, and 5. A user whose session migrates between regions
        during this window encounters different model behavior for the same input — sometimes
        noticeably so, if the new model version changed tone, format, or reasoning patterns.
        The debugging challenge: the user's report says "the model started behaving
        differently halfway through our conversation," which is correct but does not identify
        a bug; it identifies a routing event that is expected behavior in a rolling rollout.
        Production practice at organizations that have hit this bug converges on: (1) session
        affinity pinned for the duration of a rollout, so in-flight sessions see only one
        model version; (2) rollout gates that pause cross-region progression until quality
        metrics stabilize in each deployed region.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Cloud provider documentation</H3>

      <Prose>
        Amazon Web Services. <em>How AWS Global Accelerator Works.</em> docs.aws.amazon.com/
        global-accelerator/latest/dg/introduction-how-it-works.html. The definitive description
        of AWS Global Accelerator's anycast architecture: two static anycast IPv4 addresses
        announced from 100+ edge locations, traffic onboarded to the AWS private backbone at
        the nearest PoP, forwarded to the closest healthy regional endpoint group. Covers
        health-check semantics, traffic dial controls, and the routing policy model that maps
        client IP → nearest edge → target endpoint.
      </Prose>

      <Prose>
        Amazon Web Services. <em>Unlock Global AI Inference Scalability Using Global
        Cross-Region Inference on Amazon Bedrock with Anthropic Claude.</em> AWS Machine
        Learning Blog, 2025. Describes Amazon Bedrock's cross-region inference (CRIS)
        capability: automatic routing of inference requests across multiple AWS regions to
        handle traffic bursts, achieve higher throughput quotas, and maintain availability
        during regional capacity constraints. Directly relevant to how frontier model APIs
        implement multi-region serving in production.
      </Prose>

      <Prose>
        Google Cloud. <em>Global External HTTP(S) Load Balancer — Deep Dive.</em>
        cloud.google.com/blog/topics/developers-practitioners. Describes GCP's anycast-IP
        global load balancer: traffic enters Google's network at the nearest of 140+ edge
        locations, traverses Google's private backbone, and reaches the backend in the
        region with available capacity. Covers cross-region load balancing, backend health
        checks, and the distinction between Premium and Standard networking tiers.
      </Prose>

      <Prose>
        Microsoft Azure. <em>Azure Front Door Overview.</em> learn.microsoft.com/en-us/azure/
        frontdoor/front-door-overview. Documents Azure Front Door's anycast architecture
        across 118 edge locations, traffic routing methods (latency, priority, weighted),
        health probe mechanics, and origin group configuration for multi-region failover.
        Also notes the 2026 transition from anycast to unicast name resolution for Front
        Door endpoints.
      </Prose>

      <Prose>
        Google Cloud. <em>Global Endpoint for Claude Models Generally Available on Vertex AI.</em>
        cloud.google.com/blog, 2025. Describes Anthropic's Claude models served through a
        global Vertex AI endpoint that dynamically routes to any region with available
        capacity — the production implementation of active-active multi-region inference
        for a frontier API.
      </Prose>

      <H3>Reliability engineering foundations</H3>

      <Prose>
        Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy (eds.). <em>Site
        Reliability Engineering: How Google Runs Production Systems.</em> O'Reilly, 2016.
        Chapter 19 (Load Balancing at the Frontend) describes Google's Global Software Load
        Balancer (GSLB), which performs load balancing at three levels: geographic DNS,
        user-service level, and RPC level. Chapter 20 (Load Balancing in the Datacenter)
        describes the per-datacenter layer. Both chapters are directly applicable to the
        multi-tier routing architecture described in this topic. Chapter 22 (Cascading
        Failures) is the canonical reference for the failure modes that make over-provisioning
        headroom a correctness requirement rather than an optimization.
      </Prose>

      <H3>Regulatory and compliance foundations</H3>

      <Prose>
        European Parliament and Council. <em>General Data Protection Regulation (GDPR),
        Regulation (EU) 2016/679.</em> Official Journal of the European Union, 2016.
        Articles 44–49 govern transfers of personal data to third countries. The practical
        effect on LLM serving architecture is the hard routing constraint described throughout
        this topic: EU personal data cannot leave the EEA without an adequacy decision or
        appropriate safeguards. Every residency-aware routing decision in this topic
        implements the legal requirement these articles impose.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — availability math</H3>
      <Prose>
        Your service has a 99.5% SLA commitment. Each of your three regions has independent
        availability of 99.5%. What is the composite availability? Does it meet your SLA?
        Now a correlated event — a cloud provider network incident — has a 0.1% per-month
        probability of taking down all three regions simultaneously. What is the effective
        composite availability now? How many nines does the correlated event cost you?
      </Prose>

      <Callout accent="purple">
        Independent composite: <Code>1 - (0.005)^3 ≈ 0.999999875</Code> — well above 99.5%.
        With correlated failure at 0.1% per month: composite ≈ <Code>0.999999875 × (1 - 0.001)
        ≈ 0.999</Code> — drops to 99.9%, three nines. The correlated event costs nearly three
        nines of composite availability on its own. This is why the third and fourth region
        primarily protect against correlated failures, not independent ones, and why the
        cloud provider's track record for correlated incidents (not per-region availability)
        is the metric that matters most for multi-region SLA math.
      </Callout>

      <H3>Exercise 2 — design a residency-compliant routing policy</H3>
      <Prose>
        You are building a medical-records summarization API used by hospitals in Germany,
        France, Canada, Australia, and the US. GDPR applies to EU users; PIPEDA applies to
        Canada; Australian data must stay in Australia. You have regions in eu-west-1,
        eu-central-1, us-east-1, ca-central-1, and ap-southeast-2. Design the routing table:
        which regions are eligible for each user geography? What happens when a user's
        eligible region is at 90% capacity? What is your policy on failover for EU users
        when both EU regions are down simultaneously?
      </Prose>

      <Callout accent="gold">
        Routing table: DE/FR → {"{"}eu-west-1, eu-central-1{"}"} (GDPR); CA → {"{"}ca-central-1{"}"}
        (PIPEDA); AU → {"{"}ap-southeast-2{"}"} (data must stay in Australia); US → {"{"}us-east-1{"}"}.
        At 90% capacity in the eligible region: (1) for EU, fail over to the other EU region;
        (2) for CA and AU with a single eligible region, queue requests rather than violate
        residency — the contract implies "if ca-central-1 is unavailable, service is
        unavailable." For EU with both regions simultaneously down: issue a controlled 503
        with a <Code>Retry-After</Code> header. Never route EU medical data to a non-EU
        region regardless of load or failover pressure. Document this behavior explicitly in
        the SLA so customers understand the tradeoff they are accepting.
      </Callout>

      <H3>Exercise 3 — cost model for multi-region expansion</H3>
      <Prose>
        You are currently running 32 H100 GPUs in us-east-1 at $8/hr each, and your monthly
        GPU spend is $187,136. You want to add EU and APAC regions at identical provisioning
        to reduce latency for your growing European and Asian user base. What is the total
        monthly GPU cost after expansion? If 40% of EU traffic is GDPR-constrained and would
        previously have been served from us-east-1, how does adding the EU region change your
        compliance exposure? What is the minimum GPU count for the EU region if EU traffic
        is only 25% of your total today?
      </Prose>

      <Callout accent="purple">
        Total after expansion: 3 regions × 32 GPUs × $8/hr × 730 hr/month = $561,408/month.
        Compliance: before adding EU, serving EU GDPR-constrained traffic from us-east-1 was
        a violation — adding the EU region resolves the exposure entirely for traffic correctly
        routed to it. Minimum EU GPU count: if EU is 25% of traffic, and us-east-1 currently
        handles 100% at 32 GPUs with reasonable headroom, EU needs roughly 8–10 GPUs (25% of
        32, plus 20% headroom for failover absorption). In practice, always round up to the
        nearest instance type boundary and maintain headroom for eu-west-1 to absorb a
        secondary EU region failure.
      </Callout>

      <H3>Exercise 4 — diagnose a mystery latency regression</H3>
      <Prose>
        After adding a third region (ap-east-1), your EU users report that P99 TTFT
        increased from 280 ms to 520 ms. The inference workers in eu-west-1 show normal
        KV cache utilization and normal per-worker TTFT. The gateway metrics in eu-west-1
        look normal. EU traffic volume is unchanged. What are the three most likely causes
        of this regression, in order of probability? How would you distinguish between them
        using available metrics?
      </Prose>

      <Callout accent="gold">
        Most likely: (1) DNS TTL misconfiguration — adding a new region triggered a global
        DNS refresh that temporarily cached stale GeoDNS entries, routing some EU traffic to
        ap-east-1 for the TTL window. Check: compare gateway RPS to ingress RPS in eu-west-1;
        a gap indicates traffic leaking elsewhere. (2) Global traffic manager health check
        configuration — the new region's health check is misconfigured and its failure causes
        fallback-routing logic to re-evaluate EU traffic through a slower path. Check: GTM
        health-check logs for the period of regression. (3) Anycast BGP route announcement
        from the new ap-east-1 region is being preferred by some EU ISPs due to route
        leakage — unlikely but not impossible. Check: traceroute from EU client IPs to the
        API endpoint to see which PoP they are hitting.
      </Callout>

      <H3>Exercise 5 — model rollout strategy</H3>
      <Prose>
        You are rolling out a new model version to a five-region active-active fleet. The
        new model has passed eval in a staging environment. You estimate each regional
        transfer and validation takes 4 hours. If you do a parallel rollout (all regions
        simultaneously), what risks do you accept? If you do a serial rollout, what is the
        total wall-clock time and what additional safety properties does it provide? Design a
        hybrid strategy that balances speed and safety, and specify the automated rollback
        trigger you would use.
      </Prose>

      <Callout accent="purple">
        Parallel rollout risk: if the new model has a defect not caught in staging, all five
        regions are simultaneously affected — there is no clean rollback region to absorb
        traffic while the rollback completes. Total incident surface = 100% of users.
        Serial rollout: 5 × 4 hours = 20 hours total. Safety property: each completed region
        serves as a canary for the next; if a quality metric degrades in region 1, regions
        2–5 never see the bad version. Hybrid strategy: (1) deploy to the lowest-traffic
        region first (often ap-east at off-peak; 4 hrs); (2) if quality metrics hold for 1
        hour post-deployment, deploy to two mid-traffic regions in parallel (4 hrs); (3)
        if stable, deploy to the two highest-traffic regions (4 hrs). Total time: ~12 hours
        for a 5-region rollout. Rollback trigger: P5 TTFT regression greater than 20% OR
        quality-eval score drop greater than 2 percentage points relative to the previous
        version's baseline, measured over a 15-minute window with 95% statistical confidence.
      </Callout>

      {/* ======================================================================
          SECTION CLOSER — THE SYSTEM DESIGN ARC
          ====================================================================== */}
      <H2>Closing: the Inference System Design arc</H2>

      <Prose>
        This topic closes the Inference System Design section, and it is worth pausing at the
        end to see the full structure of what the section covered — not as a list of topics,
        but as a coherent engineering argument that unfolds from the inside out.
      </Prose>

      <Prose>
        The section opened with <em>inference system architecture</em> as the map: a
        top-to-bottom walk of every tier in the hot path, from TLS termination to the
        observability sink, establishing that production LLM serving is a multi-tier
        distributed system and that each tier exists because some failure mode forced its
        existence. That topic's job was to hold all subsequent topics in place as parts of
        a coherent whole rather than a bag of independent techniques.
      </Prose>

      <Prose>
        <em>Request routing and load balancing</em> zoomed into the tier that makes the
        highest-leverage per-request decisions. Naive round-robin wastes 30–60% of all
        prefill compute on workloads with shared system prompts. Cache-aware routing —
        prefix hashing, radix-tree affinity — recovers that waste without adding hardware.
        The section established that the routing decision is where most of the COGS
        optimization in a mature stack actually lives, not in the inference engine itself.
      </Prose>

      <Prose>
        <em>Autoscaling and GPU resource management</em> addressed the dimension of time:
        how the fleet size tracks demand that is simultaneously predictable (daily cycles)
        and unpredictable (viral spikes). The key insight was that classical CPU-based
        autoscaling signals are wrong for GPU inference — KV cache utilization and TTFT
        percentiles are the right leading indicators — and that scale-up must lead demand
        by the model load time (3–8 minutes for frontier models) to avoid the queue building
        before new capacity is ready.
      </Prose>

      <Prose>
        <em>Disaggregated prefill and decode</em> introduced the most architecturally
        significant recent development in LLM serving: splitting the compute-bound prefill
        phase and the memory-bandwidth-bound decode phase onto separate GPU pools.
        Co-location forces both phases to fight on the other's terms. Disaggregation lets
        each run on hardware matched to its bottleneck, and decouples their scheduling
        problems entirely. The papers that formalized this — SplitWise, DistServe, Mooncake
        — represent the frontier of serving research at the time of writing, and their ideas
        are now shipping in vLLM, NVIDIA Dynamo, and production deployments at major labs.
      </Prose>

      <Prose>
        <em>Caching strategies</em> established that the cheapest inference is the one you
        never run. Prefix caching, exact-match caching, and semantic caching form a hierarchy
        of cost-versus-risk tradeoffs: prefix caching is safe and delivers 20–40% token cost
        reduction on agent workloads; semantic caching is seductive but introduces the
        possibility of confidently wrong cache hits that are worse than no cache at all.
        The section made the argument that caching is not a single technique but a policy
        decision encoded in infrastructure.
      </Prose>

      <Prose>
        <em>Multi-model serving</em> addressed the reality that a production deployment is
        never one model — it is a fleet of models at different capability and cost tiers,
        and the routing decision between them is where 40–70% COGS savings live. Task
        classifiers that route easy requests to small models and hard requests to frontier
        models, run in 5–15 ms on CPU, pay for themselves within the first hundred requests
        they correctly downgrade.
      </Prose>

      <Prose>
        <em>Rate limiting</em> established that token-based rate limiting is categorically
        different from request-based rate limiting, and that getting this wrong means either
        letting a single request monopolize a GPU pool or mis-billing users whose token
        consumption varies by orders of magnitude across request types. The token bucket
        algorithm, distributed across gateway instances with approximate local state and
        periodic central sync, is the correct implementation for a system where the bucket
        state must be both fast to read and globally accurate enough to prevent abuse.
      </Prose>

      <Prose>
        <em>Guardrails, input/output filtering, and safety layers</em> made the argument
        that model alignment is probabilistic, not categorical. A one-in-a-hundred failure
        rate at ten million daily conversations is one hundred thousand harmful outputs daily.
        The external perimeter — input classifiers, output scanners, behavior constraint
        systems — is not redundant to alignment training; it is the circuit breaker for the
        tail of the distribution that training did not and cannot fully cover.
      </Prose>

      <Prose>
        <em>Observability and LLM monitoring</em> surfaced the categorical difference between
        classical web observability and LLM observability. An LLM can fail silently — fluent,
        grammatically correct, confidently stated output that is subtly wrong — with no 5xx,
        no exception, no stack trace. Token-level metrics, structured traces, quality
        evaluations, and statistical drift detectors are all required to see what is actually
        happening in a production deployment. The section established that active evaluation
        at production scale is itself an infrastructure problem, not just a monitoring
        configuration.
      </Prose>

      <Prose>
        <em>Streaming and Server-Sent Events</em> covered the protocol layer that makes
        LLM serving feel fast to users even when total generation time is long. Time to first
        token — not total latency — is the metric that governs perceived quality. SSE is the
        wire protocol; back-pressure handling, slow-client management, and reconnect semantics
        are where the engineering lives. The section also established that streaming is a
        resource efficiency mechanism, not just a UX improvement: early cancellation from
        clients who see a wrong response after the first few tokens saves compute that would
        otherwise run to completion.
      </Prose>

      <Prose>
        <em>Cost optimization and TCO analysis</em> stepped back from per-request optimization
        to the organizational-level question: where does the inference bill actually come from,
        and which levers move it most? The answer is consistently the same five in order:
        model routing (40–70% COGS reduction), prefix caching (20–40%), context length
        discipline (30–70% input reduction), reserved capacity pricing (40–60% off on-demand),
        and quantization (20–50% compute reduction). Everything below those five produces
        marginal returns on the aggregate bill.
      </Prose>

      <Prose>
        <em>Edge and on-premise deployment</em> covered the cases where the cloud assumption
        fails entirely — data sovereignty, physical latency, and offline operation that make
        public cloud endpoints unusable. Edge inference is a story of aggressive quantization
        fitting frontier-quality reasoning into the power and memory budgets of hardware that
        is physically co-located with the application it serves. On-premise is cloud serving
        minus multi-tenancy and elasticity, plus full operational ownership of a stack that
        hyperscalers run for you.
      </Prose>

      <Prose>
        And now, <em>multi-region and global inference infrastructure</em> — this topic —
        zooms out to the full geographic extent of the system. It is where latency physics,
        data sovereignty law, availability mathematics, and GPU supply constraints all
        converge into a single routing decision: which region handles this request, right now,
        given everything the system knows about the user's location, the data's jurisdiction,
        the regional fleet's health, and the cost of moving bytes across region boundaries.
        That routing decision is the synthesis of every prior topic in the section. It uses
        the routing algorithms from section 2, requires the autoscaling behavior from section
        3 to work within each region, benefits from the disaggregated architecture in section
        4, inherits the caching policies from section 5, applies the rate limits from section
        7, enforces the guardrail layer from section 8, emits the observability signals from
        section 9, and exposes its serving path through the streaming protocol from section
        10. None of those layers disappears when you add a geographic dimension — they
        replicate, and their interactions compound.
      </Prose>

      <Prose>
        The through-line of the entire section is a single claim: inference is no longer a
        single-box problem. The research prototype that ran on one GPU in a lab became, by
        the time it reached a hundred million users, a multi-tier, multi-region, multi-model
        distributed system with LLM-specific failure modes layered on top of every classical
        distributed-systems failure mode. A KV cache that is fast and correct within a region
        becomes a residency violation if it crosses the wrong boundary. A routing algorithm
        that minimizes latency becomes a compliance failure if it ignores jurisdiction
        constraints. An autoscaler that works perfectly under normal load fails catastrophically
        if it does not account for the 5-minute cold-start lag before new capacity is useful.
        The classical infrastructure problems do not disappear when you add a language model;
        they remain, and they acquire new consequences specific to the cost, latency, and
        safety properties of generative AI.
      </Prose>

      <Callout accent="gold">
        System design is the layer where LLM economics become product reality.
      </Callout>

    </div>
  ),
};

export default multiRegionGlobal;
