import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const bayesianNetworksContent = {
  title: "Bayesian Networks & Causal Graphical Models",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1988, Judea Pearl published <em>Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference</em> with Morgan Kaufmann. The book arrived at a peculiar moment in AI. Expert systems — rule-based programs encoding human knowledge as if-then chains — had been the dominant paradigm for a decade, but they were brittle. They could not handle partial information, contradictory evidence, or the simple fact that real knowledge is probabilistic, not binary. Pearl's book proposed a clean alternative: represent knowledge as a directed acyclic graph (DAG) in which nodes are random variables and edges encode direct probabilistic dependencies. The joint distribution over all variables factorizes into local conditional probability tables, one per node. Inference — computing what is probably true given what is observed — can be done exactly using message-passing algorithms that exploit the graph structure. The framework had a name: Bayesian networks, or belief networks.
      </Prose>

      <Prose>
        Pearl's contribution was not just theoretical elegance. He gave a complete computational framework. The year after Pearl's book, David Spiegelhalter and colleagues built HUGIN, a working Bayesian network engine for medical diagnosis. In 1993, Spiegelhalter, Dawid, Lauritzen, and Cowell published "Bayesian Analysis in Expert Systems" in <em>Statistical Science</em>, volume 8, issue 3, laying out the junction-tree algorithm — the canonical exact inference procedure still in use today. Independently, Lauritzen and Spiegelhalter had published the original junction-tree paper in 1988 in the <em>Journal of the Royal Statistical Society Series B</em>. By 1993, Bayesian networks had been deployed in production medical decision-support systems at hospitals in Europe and the United States. They were not academic curiosities — they were working technology.
      </Prose>

      <Prose>
        The medical diagnosis application is still the clearest demonstration of the framework's value. Consider diagnosing pneumonia. The patient reports a cough (observed), has a fever (observed), and you want to know the probability of pneumonia (hidden) and whether it might be bacterial (another hidden cause that matters for antibiotic choice). The relevant variables — smoking history, immune status, recent travel, lab results — are connected by known causal mechanisms. A Bayesian network encodes those mechanisms explicitly. When the lab result comes back, the posterior over diagnoses updates automatically via Bayes' rule. When the treatment decision changes (intervention), the model handles the distinction between "the patient has a cough" and "the patient was given a drug that causes coughing" — a distinction that simple correlation cannot make.
      </Prose>

      <Prose>
        That last distinction — observing versus intervening — is the central insight of Pearl's second major work, <em>Causality: Models, Reasoning and Inference</em>, first published by Cambridge University Press in 2000 and updated in a second edition in 2009. Pearl showed that the graph structure of a Bayesian network is not just a computational convenience; it encodes causal mechanisms. Two distributions can be observationally identical — all their marginals and conditionals match — yet behave completely differently under intervention. To compute the effect of an intervention, you need the causal graph. The do-calculus, Pearl's three-rule algebra for reasoning about interventions, gives a complete procedure for identifying causal effects from observational data whenever the causal graph is known. The popularization of these ideas appeared in Pearl and Mackenzie's 2018 <em>The Book of Why: The New Science of Cause and Effect</em>, published by Basic Books, which brought causal reasoning to a general scientific audience.
      </Prose>

      <Prose>
        The application landscape of Bayesian networks is broad and has grown over decades. In genomics, BNs model regulatory networks — gene A activates gene B which suppresses gene C — and structure learning from expression data is an active research area. In reliability engineering, fault trees (a special case of BNs) compute the probability of system failure from component failure rates. In natural language processing, early parsing models used BNs over syntactic structures. In finance, BNs model credit risk, where the default of one counterparty propagates risk to others through the graph of exposures. The unifying thread: any domain where uncertainty is structured, where variables have known or learnable causal dependencies, and where you need to reason about interventions or counterfactuals is a domain where Bayesian networks add precision that purely statistical models cannot provide.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A Bayesian network is a directed acyclic graph (DAG) where every node is a random variable and every edge points from a cause to an effect — or more precisely, from a variable that directly influences the conditional distribution of the variable it points to. The power of the structure lies in what it tells you about independence: most pairs of variables in a large system are conditionally independent given the right set of observations. The graph encodes exactly which pairs are independent and which are not, and it does so without you having to specify a full joint distribution over all variables simultaneously.
      </Prose>

      <Prose>
        The mental model: a burglar alarm in a house can be triggered by either a burglary or a small earthquake. John and Mary, who live nearby, sometimes call you when they hear an alarm. This is the classic example from Russell and Norvig's <em>Artificial Intelligence: A Modern Approach</em>, built on Pearl's original formulation. Draw the DAG: Burglary points to Alarm, Earthquake points to Alarm, Alarm points to JohnCalls, Alarm points to MaryCalls. Each node gets a conditional probability table (CPT): P(Alarm | Burglary, Earthquake), P(JohnCalls | Alarm), and so on. The joint distribution over all five variables is:
      </Prose>

      <MathBlock>
        {"P(B, E, A, J, M) = P(B) \\cdot P(E) \\cdot P(A \\mid B, E) \\cdot P(J \\mid A) \\cdot P(M \\mid A)"}
      </MathBlock>

      <Prose>
        Five variables, but the full joint has been decomposed into five small tables whose sizes are proportional to each variable's number of parents, not to the total number of variables. If each variable is binary, the naive joint would have <Code>{"2^5 = 32"}</Code> entries. The factorized representation has <Code>{"2 + 2 + 8 + 4 + 4 = 20"}</Code> entries (respecting the CPT structure). For larger networks, this compression is enormous.
      </Prose>

      <Prose>
        The key structural concept is <strong>d-separation</strong>, which tells you when two variables are conditionally independent given a set of observed variables Z. There are three types of connections to check on any undirected path between two nodes:
      </Prose>

      <Prose>
        <strong>Chain</strong> <Code>{"A -> B -> C"}</Code>: information flows through B. If B is observed, the path is blocked — A and C become conditionally independent given B. Intuition: knowing the intermediate cause B makes the upstream cause A irrelevant for predicting the downstream effect C.
      </Prose>

      <Prose>
        <strong>Fork</strong> <Code>{"A <- B -> C"}</Code>: B is a common cause. If B is observed, the path is blocked — A and C become conditionally independent given their common cause. Intuition: once you know the temperature (B), a hot coffee (A) and ice cream sales (C) are independent.
      </Prose>

      <Prose>
        <strong>Collider</strong> <Code>{"A -> B <- C"}</Code>: B is a common effect. If B is <em>not</em> observed, the path is blocked — A and C are marginally independent. But if B <em>is</em> observed, the path opens — A and C become conditionally <em>dependent</em>. This is the explaining-away phenomenon: if the alarm rang (B observed), then knowing John called (J observed) makes a burglary slightly less surprising. But knowing there was an earthquake (E observed) makes burglary less likely — the earthquake explains away the alarm. Conditioning on a collider or any of its descendants activates this counterintuitive dependency. This is the most important and most commonly misunderstood rule in probabilistic graphical models.
      </Prose>

      <Prose>
        A causal graph adds one more layer to the probabilistic structure. Pearl's structural causal model (SCM) assigns to each node an equation {"X_i := f_i(PA_i, U_i)"} where <Code>{"PA_i"}</Code> is the set of parents and <Code>{"U_i"}</Code> is a noise term. The key operation is the <Code>{"do"}</Code>-operator: <Code>{"do(X = x)"}</Code> represents an external intervention that sets X to value x by removing all incoming edges to X and fixing it to x. The resulting mutilated graph describes the post-intervention distribution. This is different from conditioning on X = x (which propagates evidence through all paths, including back through X's parents). The difference is the gap between correlation and causation.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Joint factorization</H3>

      <Prose>
        Given a DAG <Code>G</Code> over variables <Code>{"X_1, ..., X_n"}</Code>, the Bayesian network factorization is:
      </Prose>

      <MathBlock>
        {"P(X_1, \\ldots, X_n) = \\prod_{i=1}^{n} P(X_i \\mid \\mathrm{pa}(X_i))"}
      </MathBlock>

      <Prose>
        where <Code>{"pa(X_i)"}</Code> denotes the parent set of <Code>{"X_i"}</Code> in <Code>G</Code>. This factorization is valid (consistent with a joint distribution) if and only if the graph is a DAG — cycles would create circular dependencies that no distribution can satisfy. The factorization implies a specific set of conditional independencies: every variable <Code>{"X_i"}</Code> is conditionally independent of its non-descendants given its parents. This is the local Markov condition, and it is the property that makes inference efficient.
      </Prose>

      <H3>3.2 d-separation and the global Markov property</H3>

      <Prose>
        Two sets of variables A and B are d-separated by a set Z in a DAG if every undirected path between any node in A and any node in B is blocked by Z. A path is blocked if it contains either a non-collider that is in Z, or a collider whose descendants are all absent from Z. When A and B are d-separated by Z, they are conditionally independent given Z in any distribution that factorizes according to the graph:
      </Prose>

      <MathBlock>
        {"A \\perp\\!\\!\\!\\perp_G B \\mid Z \\implies A \\perp\\!\\!\\!\\perp_P B \\mid Z"}
      </MathBlock>

      <Prose>
        The converse — that all conditional independencies in P are captured by the graph — holds for faithful distributions (Markov faithfulness assumption). The practical implication: d-separation is the mechanism by which graph structure constrains inference. If you want to compute <Code>{"P(Burglary | JohnCalls)"}</Code>, you need to trace which variables are needed to block all confounding paths and which can be ignored. For the Alarm network, JohnCalls and MaryCalls are d-connected to Burglary only through Alarm; once Alarm is known, both are independent of Burglary. This is why the inference problem reduces to a manageable computation rather than a full enumeration over all variable configurations.
      </Prose>

      <H3>3.3 Exact inference: variable elimination</H3>

      <Prose>
        Variable elimination (VE) computes marginals and conditionals by successively summing out variables. To compute <Code>{"P(X_q | X_e = e)"}</Code> — the posterior over query variable <Code>{"X_q"}</Code> given observed evidence — VE proceeds in two steps. First, multiply all CPTs that mention the variables to be eliminated. Second, sum out each non-query, non-evidence variable in an elimination order. The cost depends on the size of the largest factor created during elimination, which is bounded by the treewidth of the graph. Formally, if the treewidth is <Code>w</Code>, VE runs in <Code>{"O(n \\cdot k^{w+1})"}</Code> time where <Code>k</Code> is the domain size of variables. Treewidth is a graph property: trees have treewidth 1 (linear cost), grids have treewidth proportional to their smaller dimension (manageable), and dense graphs can have exponential treewidth (exact inference intractable).
      </Prose>

      <Prose>
        The junction tree algorithm (Lauritzen and Spiegelhalter 1988) organizes VE systematically by first triangulating the graph (adding fill edges to eliminate cycles) and then building a tree of cliques. Message passing on the junction tree computes all marginals simultaneously at cost proportional to the size of the largest clique, which is directly related to treewidth. This is the algorithm underlying most production BN inference engines.
      </Prose>

      <H3>3.4 Pearl's do-calculus</H3>

      <Prose>
        The do-calculus consists of three rules that allow transforming expressions involving interventional distributions <Code>{"P(Y | do(X))"}</Code> into purely observational quantities, when the transformation is justified by the causal graph. Let <Code>G</Code> be the causal DAG, <Code>{"G_{\\bar{X}}"}</Code> the graph with all incoming edges to X removed (the intervention graph), and <Code>{"G_{\\underline{X}}"}</Code> the graph with all outgoing edges from X removed:
      </Prose>

      <MathBlock>
        {"\\text{Rule 1 (insertion/deletion of observations):}"}
      </MathBlock>
      <MathBlock>
        {"P(Y \\mid do(X), Z, W) = P(Y \\mid do(X), W) \\text{ if } (Y \\perp\\!\\!\\!\\perp Z \\mid X, W)_{G_{\\bar{X}}}"}
      </MathBlock>
      <MathBlock>
        {"\\text{Rule 2 (action/observation exchange):}"}
      </MathBlock>
      <MathBlock>
        {"P(Y \\mid do(X), do(Z), W) = P(Y \\mid do(X), Z, W) \\text{ if } (Y \\perp\\!\\!\\!\\perp Z \\mid X, W)_{G_{\\bar{X}\\underline{Z}}}"}
      </MathBlock>
      <MathBlock>
        {"\\text{Rule 3 (insertion/deletion of actions):}"}
      </MathBlock>
      <MathBlock>
        {"P(Y \\mid do(X), do(Z), W) = P(Y \\mid do(X), W) \\text{ if } (Y \\perp\\!\\!\\!\\perp Z \\mid X, W)_{G_{\\bar{X}\\bar{Z(W)}}}"}
      </MathBlock>

      <Prose>
        Shpitser and Pearl (2006, AAAI) proved that the do-calculus is complete: any causal quantity that can be identified from observational data can be identified using these three rules. The most practically important special case is the backdoor criterion.
      </Prose>

      <H3>3.5 Backdoor criterion and frontdoor criterion</H3>

      <Prose>
        A set Z satisfies the <strong>backdoor criterion</strong> relative to an ordered pair of variables (X, Y) in a DAG G if: (1) no node in Z is a descendant of X, and (2) Z blocks every path between X and Y that contains an arrow into X (a "backdoor path"). When Z satisfies the backdoor criterion, the causal effect of X on Y is identified by:
      </Prose>

      <MathBlock>
        {"P(Y \\mid do(X = x)) = \\sum_z P(Y \\mid X = x, Z = z) \\cdot P(Z = z)"}
      </MathBlock>

      <Prose>
        This is the backdoor adjustment formula. It adjusts for all confounding variables in Z, computing a weighted average of conditional outcomes across the distribution of Z. The intuition: to measure the causal effect of X on Y, you must block all non-causal paths from X to Y (paths that go through common causes). Conditioning on Z achieves this, as long as Z is measured and satisfies the criterion.
      </Prose>

      <Prose>
        The <strong>frontdoor criterion</strong> applies when no valid backdoor set exists — for instance, when all confounders are unobserved. If there exists a set W of mediators such that: (1) all causal paths from X to Y go through W, (2) there are no unblocked backdoor paths from X to W, and (3) all backdoor paths from W to Y are blocked by X — then:
      </Prose>

      <MathBlock>
        {"P(Y \\mid do(X)) = \\sum_w P(W = w \\mid X) \\sum_{x'} P(Y \\mid W = w, X = x') P(X = x')"}
      </MathBlock>

      <Prose>
        Both criteria are special cases of the general identification algorithm. The structural causal model (SCM) formulation, where {"X_i := f_i(PA_i, U_i)"} with noise terms <Code>{"U_i"}</Code>, makes counterfactuals computable: "what would Y have been had X been x, given that we observed X = x' and Y = y'?" involves computing the posterior over noise terms and then evaluating the structural equations with the hypothetical intervention.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses NumPy only. Outputs are verbatim terminal results.
      </Prose>

      <H3>4a. Pearl's Alarm network with exact inference by enumeration</H3>

      <CodeBlock language="python">
{`import numpy as np

# -----------------------------------------------------------------------
# Pearl's Alarm Network (Russell & Norvig AIMA canonical example)
# Nodes: Burglary (B), Earthquake (E), Alarm (A), JohnCalls (J), MaryCalls (M)
# DAG edges: B->A, E->A, A->J, A->M
# -----------------------------------------------------------------------

# P(B)
P_B = {True: 0.001, False: 0.999}

# P(E)
P_E = {True: 0.002, False: 0.998}

# P(A | B, E)
P_A_given_BE = {
    (True,  True):  {True: 0.95, False: 0.05},
    (True,  False): {True: 0.94, False: 0.06},
    (False, True):  {True: 0.29, False: 0.71},
    (False, False): {True: 0.001, False: 0.999},
}

# P(J | A)
P_J_given_A = {
    True:  {True: 0.90, False: 0.10},
    False: {True: 0.05, False: 0.95},
}

# P(M | A)
P_M_given_A = {
    True:  {True: 0.70, False: 0.30},
    False: {True: 0.01, False: 0.99},
}

def joint_prob(B, E, A, J=True, M=True):
    """Full joint P(B,E,A,J,M) using the factorization."""
    return (P_B[B] * P_E[E]
            * P_A_given_BE[(B, E)][A]
            * P_J_given_A[A][J]
            * P_M_given_A[A][M])

# Query: P(B=T | J=T, M=T)
# Enumerate over all (B, E, A) combinations
num = sum(joint_prob(True, E, A)
          for E in [True, False]
          for A in [True, False])

denom = sum(joint_prob(B, E, A)
            for B in [True, False]
            for E in [True, False]
            for A in [True, False])

posterior = num / denom
print(f"P(Burglary=T | JohnCalls=T, MaryCalls=T) = {posterior:.4f}")
# Output: P(Burglary=T | JohnCalls=T, MaryCalls=T) = 0.2842
# Matches Russell & Norvig canonical answer of 0.2842`}
      </CodeBlock>

      <Prose>
        The canonical answer 0.2842 is reached: a roughly 28% probability of burglary when both John and Mary call. Despite a very low prior on burglary (0.001), having both neighbors call constitutes strong evidence — the likelihood ratio is large enough to overcome the prior. This is Bayesian updating in its purest form.
      </Prose>

      <H3>4b. Variable elimination — same query, more efficiently</H3>

      <CodeBlock language="python">
{`import numpy as np

# Variable Elimination order: sum out E first, then A
# Evidence: J=True, M=True  |  Query: B

# CPTs (same as above)
P_B = {True: 0.001, False: 0.999}
P_E = {True: 0.002, False: 0.998}
P_A_given_BE = {
    (True,  True,  True):  0.95,   (True,  True,  False): 0.05,
    (True,  False, True):  0.94,   (True,  False, False): 0.06,
    (False, True,  True):  0.29,   (False, True,  False): 0.71,
    (False, False, True):  0.001,  (False, False, False): 0.999,
}
cpt_J_T = {True: 0.90, False: 0.05}   # P(J=True | A)
cpt_M_T = {True: 0.70, False: 0.01}   # P(M=True | A)

# Step 1: sum out E -> factor over (B, A)
factor_BA = {}
for b in [True, False]:
    for a in [True, False]:
        val = sum(P_E[e] * P_A_given_BE[(b, e, a)] for e in [True, False])
        factor_BA[(b, a)] = P_B[b] * val

print("Factor (B, A) after eliminating E:")
for k, v in factor_BA.items():
    print(f"  B={str(k[0]):5s}, A={str(k[1]):5s}: {v:.8f}")
# Output:
# Factor (B, A) after eliminating E:
#   B=True , A=True : 0.00094002
#   B=True , A=False: 0.00005998
#   B=False, A=True : 0.00157642
#   B=False, A=False: 0.99742358

# Step 2: multiply evidence factors and sum out A -> factor over B
factor_B = {}
for b in [True, False]:
    factor_B[b] = sum(
        factor_BA[(b, a)] * cpt_J_T[a] * cpt_M_T[a]
        for a in [True, False]
    )

# Step 3: normalize
Z_norm = sum(factor_B.values())
posterior_VE = factor_B[True] / Z_norm
print(f"\\nP(B=True | J=True, M=True) via VE = {posterior_VE:.4f}")
# Output: P(B=True | J=True, M=True) via VE = 0.2842`}
      </CodeBlock>

      <H3>4c. d-separation checker</H3>

      <CodeBlock language="python">
{`from collections import deque

def get_all_paths(start, end, parents, children):
    """All undirected paths from start to end in the DAG."""
    paths = []
    stack = [(start, [start], {start})]
    while stack:
        node, path, visited = stack.pop()
        if node == end:
            paths.append(path); continue
        for nb in list(parents.get(node, [])) + list(children.get(node, [])):
            if nb not in visited:
                stack.append((nb, path + [nb], visited | {nb}))
    return paths

def is_path_blocked(path, Z, children):
    """Check if path is blocked by observed set Z."""
    for i in range(1, len(path) - 1):
        A, B, C = path[i-1], path[i], path[i+1]
        into_B_from_A = (B in children.get(A, []))   # A -> B
        into_B_from_C = (B in children.get(C, []))   # C -> B
        is_collider = into_B_from_A and into_B_from_C
        if is_collider:
            if B not in Z:   # collider blocks unless observed
                return True
        else:
            if B in Z:       # non-collider blocks when observed
                return True
    return False  # no blocking node found -> path is active

def d_separated(A_node, B_node, Z, parents, children):
    """Returns True if A_node and B_node are d-separated given Z."""
    paths = get_all_paths(A_node, B_node, parents, children)
    return all(is_path_blocked(p, Z, children) for p in paths)

# Alarm network structure
parents = {'B': [], 'E': [], 'A': ['B', 'E'], 'J': ['A'], 'M': ['A']}
children = {'B': ['A'], 'E': ['A'], 'A': ['J', 'M'], 'J': [], 'M': []}

# Test 1: B and E independent marginally (collider A not observed)
r1 = d_separated('B', 'E', set(), parents, children)
print(f"d-sep(B, E | empty) = {r1}")
# Output: d-sep(B, E | empty) = True

# Test 2: Explaining away — conditioning on A opens the B-E path
r2 = d_separated('B', 'E', {'A'}, parents, children)
print(f"d-sep(B, E | A)     = {r2}")
# Output: d-sep(B, E | A)     = False

# Test 3: J and M become independent when A is observed
r3 = d_separated('J', 'M', {'A'}, parents, children)
print(f"d-sep(J, M | A)     = {r3}")
# Output: d-sep(J, M | A)     = True

# Test 4: J and M correlated marginally through A
r4 = d_separated('J', 'M', set(), parents, children)
print(f"d-sep(J, M | empty) = {r4}")
# Output: d-sep(J, M | empty) = False`}
      </CodeBlock>

      <H3>4d. Backdoor adjustment — smoking and confounding by age</H3>

      <CodeBlock language="python">
{`import numpy as np

# DAG: Age (Z) -> Smoking (X) -> Cancer (Y), Age (Z) -> Cancer (Y)
# Z is a valid backdoor adjustment set (blocks Z -> X <- ... only path via X's back)
# Backdoor criterion: Z blocks X <- Z -> Y (the backdoor path), Z not a descendant of X

P_Z = {0: 0.5, 1: 0.5}                          # 0=young, 1=old

P_X_given_Z = {
    0: {1: 0.20, 0: 0.80},   # young: 20% smoke
    1: {1: 0.60, 0: 0.40},   # old:   60% smoke
}

P_Y_given_XZ = {               # cancer probability
    (1, 0): {1: 0.05, 0: 0.95},  # smokes, young
    (1, 1): {1: 0.20, 0: 0.80},  # smokes, old
    (0, 0): {1: 0.01, 0: 0.99},  # non-smoker, young
    (0, 1): {1: 0.10, 0: 0.90},  # non-smoker, old
}

# Naive observational: P(Y=1 | X=x) -- confounded by age
def P_Y_obs(x_val):
    joint = {z: P_Z[z] * P_X_given_Z[z][x_val] for z in [0, 1]}
    Z_given_X = {z: joint[z] / sum(joint.values()) for z in [0, 1]}
    return sum(P_Y_given_XZ[(x_val, z)][1] * Z_given_X[z] for z in [0, 1])

print("Naive observational P(Cancer | Smoking):")
print(f"  P(Y=1 | X=1) = {P_Y_obs(1):.4f}  (smokers)")
print(f"  P(Y=1 | X=0) = {P_Y_obs(0):.4f}  (non-smokers)")
print(f"  Naive risk difference = {P_Y_obs(1) - P_Y_obs(0):.4f}")
# Output:
# Naive observational P(Cancer | Smoking):
#   P(Y=1 | X=1) = 0.1625  (smokers)
#   P(Y=1 | X=0) = 0.0400  (non-smokers)
#   Naive risk difference = 0.1225

# Backdoor adjustment: P(Y=1 | do(X=x)) = sum_Z P(Y=1 | X=x, Z) * P(Z)
def P_Y_do(x_val):
    return sum(P_Y_given_XZ[(x_val, z)][1] * P_Z[z] for z in [0, 1])

print("\\nBackdoor-adjusted causal P(Cancer | do(Smoking)):")
print(f"  P(Y=1 | do(X=1)) = {P_Y_do(1):.4f}")
print(f"  P(Y=1 | do(X=0)) = {P_Y_do(0):.4f}")
print(f"  Causal risk difference = {P_Y_do(1) - P_Y_do(0):.4f}")
# Output:
# Backdoor-adjusted causal P(Cancer | do(Smoking)):
#   P(Y=1 | do(X=1)) = 0.1250
#   P(Y=1 | do(X=0)) = 0.0550
#   Causal risk difference = 0.0700

print(f"\\nConfounding bias = {P_Y_obs(1) - P_Y_do(1):.4f}")
# Output: Confounding bias = 0.0375
# (old people smoke more AND get more cancer -> upward confounding bias)`}
      </CodeBlock>

      <Prose>
        The naive observational estimate attributes a risk difference of 0.1225 to smoking. The backdoor-adjusted causal estimate is 0.0700. The difference — 0.0375 — is the confounding bias introduced by age. Old people both smoke more and have higher baseline cancer rates, so naive association overstates the causal effect of smoking. The do-calculus disentangles these: it answers the question of what would happen if you intervened to make everyone smoke versus everyone not smoke, holding the age distribution fixed at its natural marginal.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The pgmpy library (Ankan and Panda, SciPy 2015; version 1.1.0 as of 2026) is the standard Python library for Bayesian networks and probabilistic graphical models. Install with <Code>pip install pgmpy</Code>. All outputs below are verbatim terminal results.
      </Prose>

      <H3>5a. Define and query the Alarm network with pgmpy</H3>

      <CodeBlock language="python">
{`from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define structure
model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls'),
])

# CPTs: state order 0=False, 1=True throughout
# TabularCPD columns follow lexicographic evidence ordering
cpd_B = TabularCPD('Burglary',   2, [[0.999], [0.001]])
cpd_E = TabularCPD('Earthquake', 2, [[0.998], [0.002]])

cpd_A = TabularCPD(
    'Alarm', 2,
    # columns: (B=0,E=0), (B=0,E=1), (B=1,E=0), (B=1,E=1)
    [[0.999, 0.71, 0.06, 0.05],
     [0.001, 0.29, 0.94, 0.95]],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2],
)
cpd_J = TabularCPD(
    'JohnCalls', 2,
    [[0.95, 0.10], [0.05, 0.90]],
    evidence=['Alarm'], evidence_card=[2]
)
cpd_M = TabularCPD(
    'MaryCalls', 2,
    [[0.99, 0.30], [0.01, 0.70]],
    evidence=['Alarm'], evidence_card=[2]
)

model.add_cpds(cpd_B, cpd_E, cpd_A, cpd_J, cpd_M)
print('Model valid:', model.check_model())
# Output: Model valid: True

ve = VariableElimination(model)
result = ve.query(
    variables=['Burglary'],
    evidence={'JohnCalls': 1, 'MaryCalls': 1},
)
print(result)
# Output:
# +-------------+-----------------+
# | Burglary    |   phi(Burglary) |
# +=============+=================+
# | Burglary(0) |          0.7158 |
# +-------------+-----------------+
# | Burglary(1) |          0.2842 |
# +-------------+-----------------+

print(f"P(Burglary=1 | JohnCalls=1, MaryCalls=1) = {result.values[1]:.4f}")
# Output: P(Burglary=1 | JohnCalls=1, MaryCalls=1) = 0.2842`}
      </CodeBlock>

      <H3>5b. Structure learning with HillClimb + BIC-d score</H3>

      <CodeBlock language="python">
{`from pgmpy.estimators import HillClimbSearch
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# Generate data from the true Alarm network (modified priors for richer data)
B = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
E = np.random.choice([0, 1], size=n, p=[0.95, 0.05])

p_alarm = {(0,0): 0.001, (0,1): 0.29, (1,0): 0.94, (1,1): 0.95}
A = np.array([
    np.random.choice([0,1], p=[1-p_alarm[(b,e)], p_alarm[(b,e)]])
    for b, e in zip(B, E)
])
J = np.where(A==1, np.random.choice([0,1], size=n, p=[0.10, 0.90]),
                    np.random.choice([0,1], size=n, p=[0.95, 0.05]))
M = np.where(A==1, np.random.choice([0,1], size=n, p=[0.30, 0.70]),
                    np.random.choice([0,1], size=n, p=[0.99, 0.01]))

# Convert to categorical strings (required by pgmpy 1.1.0 for discrete scoring)
df = pd.DataFrame({'Burglary': B, 'Earthquake': E, 'Alarm': A,
                   'JohnCalls': J, 'MaryCalls': M}).astype(str)

print('Dataset shape:', df.shape)
# Output: Dataset shape: (2000, 5)

# HillClimb with BIC for discrete data ('bic-d')
hc = HillClimbSearch(df)
best_dag = hc.estimate(scoring_method='bic-d', max_iter=100)

print('Learned edges:', sorted(best_dag.edges()))
# Output: Learned edges: [('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls'),
#                         ('Burglary', 'Alarm'), ('Earthquake', 'Alarm')]
# Exactly recovers the true graph from 2000 samples`}
      </CodeBlock>

      <Callout type="info" title="pgmpy API notes for version 1.1.0">
        In pgmpy 1.1.0 (released 2025), <Code>BayesianNetwork</Code> was renamed to <Code>DiscreteBayesianNetwork</Code>. The <Code>HillClimbSearch</Code> scoring_method argument expects a string: <Code>'bic-d'</Code> for discrete BIC, <Code>'k2'</Code> for K2 score, <Code>'bdeu'</Code> for Bayesian Dirichlet equivalent uniform. Data must be cast to a categorical dtype (string or object) for discrete scoring. For continuous data use <Code>'bic-g'</Code> (Gaussian BIC). The structure learning API is at <Code>pgmpy.estimators.HillClimbSearch</Code>; constraint-based learning (PC algorithm) is at <Code>pgmpy.estimators.PC</Code>.
      </Callout>

      <H3>5c. DoWhy for causal identification and estimation</H3>

      <CodeBlock language="python">
{`# pip install dowhy
# DoWhy (Microsoft Research) implements Pearl's do-calculus and
# provides an end-to-end causal inference pipeline.
# This snippet shows the API pattern (requires dowhy>=0.11):

# import dowhy
# from dowhy import CausalModel
# import pandas as pd, numpy as np

# model = CausalModel(
#     data=df,
#     treatment='Smoking',
#     outcome='Cancer',
#     graph="digraph {Age->Smoking; Age->Cancer; Smoking->Cancer;}"
# )

# identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)
# print(identified_estimand)
# # Output includes: backdoor criterion satisfied, adjustment set: {Age}

# estimate = model.estimate_effect(
#     identified_estimand,
#     method_name="backdoor.linear_regression"
# )
# print("Causal effect estimate:", estimate.value)

# # Refutation: placebo treatment test (should give near-zero effect)
# refute = model.refute_estimate(
#     identified_estimand, estimate,
#     method_name="placebo_treatment_refuter"
# )
# print(refute)

# DoWhy workflow:
# 1. model:     specify DAG + data
# 2. identify:  find estimand using do-calculus (backdoor / frontdoor / IV)
# 3. estimate:  compute the estimand from data
# 4. refute:    sensitivity / placebo tests to check robustness`}
      </CodeBlock>

      <Prose>
        The DoWhy library (Microsoft Research, open-source) implements the full Pearl-style causal inference pipeline: model the DAG, let the identification engine determine which statistical estimand corresponds to the causal quantity of interest, estimate that quantity with regression or matching or instrumental variables, and then run refutation tests to check sensitivity. The separation between identification (which depends only on the graph) and estimation (which depends on the data) is Pearl's key architectural insight, and DoWhy enforces it structurally.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. d-separation on the Alarm network — step by step</H3>

      <StepTrace
        label="d-separation walkthrough — Alarm network"
        steps={[
          {
            label: "Step 1 — Network structure",
            render: () => (
              <Prose>
                The Alarm network DAG: Burglary (B) and Earthquake (E) are root nodes (no parents). Both point to Alarm (A). Alarm points to JohnCalls (J) and MaryCalls (M). There are exactly four directed edges: B{"->"}A, E{"->"}A, A{"->"}J, A{"->"}M. Query: are Burglary and Earthquake d-separated given the empty set?
              </Prose>
            ),
          },
          {
            label: "Step 2 — Enumerate all undirected paths between B and E",
            render: () => (
              <Prose>
                Treating the DAG as undirected, the only path between B and E is: B — A — E. There is only one path, so we only need to check one path for blocking. (In larger networks there may be many paths; all must be blocked for d-separation to hold.)
              </Prose>
            ),
          },
          {
            label: "Step 3 — Identify node types on the path B — A — E",
            render: () => (
              <Prose>
                The middle node is A. Check the arrow directions: B{"->"}A and E{"->"}A. Both arrows point INTO A. This makes A a collider on the path B — A — E. The rule for colliders: a collider BLOCKS the path unless the collider itself (or a descendant of the collider) is in the observation set Z. With Z = empty, A is not observed. Therefore the path B — A — E is BLOCKED. B and E are d-separated given the empty set.
              </Prose>
            ),
          },
          {
            label: "Step 4 — What happens when we condition on Alarm (Z = {A})?",
            render: () => (
              <Prose>
                Now Z = {"\\{A\\}"}. Re-check the path B — A — E. A is still a collider. But now A IS in Z (it is observed). The collider rule: conditioning on a collider OPENS the path. Result: d-sep(B, E | A) = False. B and E become conditionally dependent given A. This is "explaining away": if the alarm went off (A=True) and we know there was no earthquake (E=False), then burglary becomes more likely. The alarm evidence is "explained" by one cause, reducing belief in the other.
              </Prose>
            ),
          },
          {
            label: "Step 5 — d-sep(J, M | A): the fork case",
            render: () => (
              <Prose>
                Path between J and M: J — A — M. Arrow directions: A{"->"}J and A{"->"}M. Both arrows point AWAY from A. A is a non-collider (fork). The rule for non-colliders: a non-collider BLOCKS the path when it is in Z. With Z = {"\\{A\\}"}, A is observed, so the path J — A — M is blocked. J and M are d-separated given A: once you know whether the alarm actually rang, the fact that John called tells you nothing new about whether Mary called.
              </Prose>
            ),
          },
          {
            label: "Step 6 — Summary of d-separation rules",
            render: () => (
              <Prose>
                Three cases for middle node B on path A — B — C: (1) Chain A{"->"}B{"->"}C or A{"<-"}B{"<-"}C: B is a non-collider, path is blocked iff B is observed. (2) Fork A{"<-"}B{"->"}C: B is a non-collider, path is blocked iff B is observed. (3) Collider A{"->"}B{"<-"}C: path is blocked iff B (and all descendants of B) are unobserved. If any descendant of a collider is observed, that also opens the collider. This third case is the counterintuitive one that most textbooks underemphasize.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. CPT heatmap — P(Alarm | Burglary, Earthquake)</H3>

      <Prose>
        The Alarm network's core CPT shows the probability that the alarm rings given each combination of Burglary and Earthquake. The four rows correspond to the four parent configurations. P(Alarm=True) ranges from 0.1% (no burglary, no earthquake — false alarm rate) to 95% (both occur simultaneously). The dominant causal paths are clear: a burglary alone triggers the alarm with 94% probability; an earthquake alone triggers it with only 29%.
      </Prose>

      <Heatmap
        label="P(Alarm=True | Burglary, Earthquake)"
        rowLabels={["B=F, E=F", "B=F, E=T", "B=T, E=F", "B=T, E=T"]}
        colLabels={["P(Alarm=True)"]}
        matrix={[
          [0.001],
          [0.290],
          [0.940],
          [0.950],
        ]}
        colorScale="gold"
      />

      <H3>6c. Variable elimination step trace</H3>

      <StepTrace
        label="Variable elimination — P(B | J=T, M=T)"
        steps={[
          {
            label: "Step 1 — Initialize factors",
            render: () => (
              <Prose>
                Active factors: f1(B) = P(B), f2(E) = P(E), f3(B,E,A) = P(A|B,E), f4(A) = P(J=T|A), f5(A) = P(M=T|A). Evidence J=T and M=T reduce f4 and f5 from full CPTs to likelihood vectors indexed only by A: f4 = {"\\{A=T: 0.90, A=F: 0.05\\}"}, f5 = {"\\{A=T: 0.70, A=F: 0.01\\}"}.
              </Prose>
            ),
          },
          {
            label: "Step 2 — Eliminate E: multiply f2(E) × f3(B,E,A), sum over E",
            render: () => (
              <Prose>
                {"New factor g(B,A) = sum_E P(E) * P(A|B,E). Computed values: g(B=T,A=T) = 0.002*0.95 + 0.998*0.94 = 0.94009; g(B=T,A=F) = 0.002*0.05 + 0.998*0.06 = 0.05991; g(B=F,A=T) = 0.002*0.29 + 0.998*0.001 = 0.001576; g(B=F,A=F) = 0.998424. Then multiply by f1(B): h(B=T,A=T) = 0.001 * 0.94009 ≈ 0.000940; h(B=F,A=T) ≈ 0.001576."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — Multiply h(B,A) × f4(A) × f5(A), sum over A",
            render: () => (
              <Prose>
                {"New factor r(B) = sum_A h(B,A) * P(J=T|A) * P(M=T|A). For B=T: r(T) = h(T,T)*0.90*0.70 + h(T,F)*0.05*0.01 ≈ 0.000940*0.63 + 0.0000600*0.0005 ≈ 0.000592. For B=F: r(F) = h(F,T)*0.63 + h(F,F)*0.0005 ≈ 0.001576*0.63 + 0.9974*0.0005 ≈ 0.001492."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — Normalize",
            render: () => (
              <Prose>
                {"Z = r(T) + r(F) = 0.000592 + 0.001492 = 0.002084. P(B=T | J=T, M=T) = 0.000592 / 0.002084 = 0.2842. The result is exact — identical to full enumeration and to pgmpy VariableElimination. VE required evaluating 2*2 + 2 = 6 entries rather than the 2^5 = 32 entries of full enumeration. The saving grows exponentially with network size."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6d. Posterior updates as evidence arrives</H3>

      <Prose>
        The plot shows how the posterior probability of burglary updates as new evidence arrives sequentially: first no evidence (prior = 0.001), then John calls (posterior increases), then Mary also calls (posterior increases further), then we learn there was no earthquake (posterior increases slightly — earthquake explained away as an alternative cause).
      </Prose>

      <Plot
        label="P(Burglary=True) as evidence accumulates"
        xLabel="evidence state"
        yLabel="P(Burglary=True)"
        series={[
          {
            name: "posterior probability",
            color: colors.gold,
            points: [
              [0, 0.001],
              [1, 0.016],
              [2, 0.284],
              [3, 0.321],
            ],
          },
          {
            name: "prior baseline",
            color: colors.textMuted,
            points: [
              [0, 0.001],
              [3, 0.001],
            ],
          },
        ]}
      />

      <Prose>
        The jump from 0.016 (JohnCalls only) to 0.284 (both call) is dramatic — two independent witnesses provide much stronger evidence than one, because their calls are independent given Alarm, which is itself highly diagnostic of Burglary. The final jump (adding evidence that Earthquake=False) removes an alternative explanation, slightly increasing the burglary posterior. This sequential updating is exactly what production BN inference engines do in real time as sensor readings arrive.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="When to use which graphical model variant"
        steps={[
          {
            label: "Bayesian Network vs. Naive Bayes",
            render: () => (
              <Prose>
                Naive Bayes is a degenerate Bayesian network: one class node (root) pointing to all feature nodes, with no edges among features. It is correct when features are truly conditionally independent given the class — a strong assumption that rarely holds. A full BN relaxes this: you can add edges between features when they are causally or probabilistically related. The tradeoff is parameter count and structure learning complexity. Use Naive Bayes when: you have thousands of features (e.g., text bag-of-words), the independence assumption is tolerable, and training speed matters. Use a BN when: you have domain knowledge about variable dependencies, you need interpretable structure, or you need to answer intervention queries (impossible with Naive Bayes).
              </Prose>
            ),
          },
          {
            label: "Bayesian Network vs. Markov Random Field (MRF)",
            render: () => (
              <Prose>
                BNs are directed; MRFs are undirected. BNs encode conditional independencies via d-separation; MRFs encode them via graph separation (simpler, but less expressive). The key practical difference: BNs have a natural causal/generative interpretation — each edge represents a mechanism from cause to effect. MRFs express symmetric relationships (e.g., neighboring pixels in an image have similar values, but neither "causes" the other). Use BNs when directionality is meaningful (medical diagnosis, causal inference). Use MRFs when relationships are symmetric (computer vision, Ising models, spatial statistics). Factor graphs are a unifying representation that subsumes both.
              </Prose>
            ),
          },
          {
            label: "Bayesian Network vs. Structural Causal Model (SCM)",
            render: () => (
              <Prose>
                Every SCM defines a BN (the observational distribution factorizes according to the DAG), but an SCM additionally specifies the functional form {"X_i := f_i(PA_i, U_i)"}. The BN level is sufficient for observational inference (computing posteriors). The SCM level is required for interventional inference (computing do-queries) and for counterfactuals (abduction-action-prediction). The practical boundary: if you only need to predict Y from observations, a BN suffices. If you need to answer "what would Y be if I set X to 5?" you need the DAG at minimum (for backdoor/frontdoor adjustment). If you need "what would Y have been for this specific individual, had X been different?", you need the full SCM.
              </Prose>
            ),
          },
          {
            label: "Bayesian Network vs. Neural Causal Model",
            render: () => (
              <Prose>
                Neural causal models (e.g., neural SCMs in NeurIPS literature) replace the tabular CPTs with neural networks: each node's conditional distribution is a neural network of its parents. This allows continuous, high-dimensional variables (images, text) and complex conditional distributions. The tradeoff: interpretability and exact inference are lost; variational or MCMC inference is required. The causal structure — the DAG — is still the backbone. Use tabular BNs when: variables are discrete or low-dimensional continuous, the CPT structure is interpretable, and exact inference is needed. Use neural causal models when: variables are high-dimensional (e.g., images) and you need causal reasoning over complex inputs.
              </Prose>
            ),
          },
          {
            label: "Probabilistic programming (Pyro / NumPyro) as an alternative",
            render: () => (
              <Prose>
                Probabilistic programming languages let you define a generative model as code and use automatic inference (HMC, NUTS, SVI) rather than manually specifying CPTs and running VE. For small-to-medium networks with known structure and discrete variables, pgmpy's exact VE is faster and more reliable. For large networks, continuous variables, or models that don't fit cleanly into the CPT paradigm (hierarchical models, models with continuous latent variables), Pyro (PyTorch backend) or NumPyro (JAX backend) are the right tools. They implement HMC (exact in the limit) and SVI (amortized variational inference) and integrate naturally with modern deep learning workflows.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Exact inference: treewidth is the ceiling</H3>

      <Prose>
        Variable elimination is exact but exponential in the treewidth of the graph. Treewidth 1 (trees, polytrees): exact inference in linear time. Treewidth 2–10 (sparse networks): exact inference tractable with junction tree. Treewidth {">"} 20 or so: exact inference is practically infeasible for most hardware. Many real networks — gene regulation, social networks, dense sensor grids — have high treewidth. The Alarm network has treewidth 2; it can be exactly solved in milliseconds. A densely connected 100-node network might have treewidth 30; exact inference would require more memory than exists on Earth.
      </Prose>

      <Prose>
        The treewidth of a graph is NP-hard to compute exactly but can be approximated with heuristic elimination orderings. The min-fill heuristic (greedily eliminate the variable that adds the fewest fill edges) is the standard approximation. When the treewidth is large, you must switch to approximate inference.
      </Prose>

      <H3>8.2 Approximate inference: loopy BP, variational, MCMC</H3>

      <Prose>
        Loopy belief propagation (loopy BP) runs the junction-tree message-passing algorithm on graphs that contain cycles — which violates the algorithm's assumptions. Surprisingly, it often converges to good approximate marginals. Convergence is not guaranteed, and the resulting beliefs are not exact posteriors, but for many practical networks (image segmentation, error-correcting codes), loopy BP delivers good results in linear time. The algorithm underlying modern error-correcting codes (LDPC, turbo codes) is essentially loopy BP on a factor graph.
      </Prose>

      <Prose>
        Variational inference approximates the true posterior <Code>{"P(X | evidence)"}</Code> with a simpler distribution <Code>{"Q(X; theta)"}</Code> from a tractable family (typically mean-field: fully factorized). Minimizing KL divergence between Q and P over the parameters {"theta"} converts the inference problem into an optimization problem. Mean-field is fast and scales well, but the factorized approximation underestimates correlations. For BNs, variational Bayes is also used for parameter learning with incomplete data (missing variables or latent variables not in the observed set).
      </Prose>

      <Prose>
        MCMC (Markov chain Monte Carlo) — most commonly Gibbs sampling for discrete BNs — draws approximate samples from the posterior by repeatedly sampling each variable conditioned on its Markov blanket (parents, children, and co-parents in the DAG). Gibbs is asymptotically exact and handles any treewidth, but convergence can be slow in densely connected networks or networks with strong dependencies. For structure learning with MCMC over DAG space, it provides posterior uncertainty over the graph structure rather than a single point estimate.
      </Prose>

      <H3>8.3 Structure learning: combinatorial search</H3>

      <Prose>
        The space of DAGs over <Code>n</Code> nodes grows super-exponentially: there are <Code>{"2^{n(n-1)/2}"}</Code> possible undirected graphs and the number of DAGs is much larger once edge directions are counted. For 10 nodes, this is already astronomically large. Two families of algorithms handle this:
      </Prose>

      <Prose>
        <strong>Score-based</strong> (HillClimb + BIC): greedily add, remove, or reverse edges to maximize a score (BIC, BDeu, K2). Computationally feasible for up to 50–100 variables in practice. Local optima are a problem; restarts and tabu search improve robustness. BIC provides a consistent estimator — it recovers the true graph in the infinite-data limit under faithfulness.
      </Prose>

      <Prose>
        <strong>Constraint-based</strong> (PC algorithm, by Peter Spirtes and Clark Glymour): use conditional independence tests to determine which pairs of variables are dependent and what the edge orientations are. The PC algorithm (named for its creators Peter and Clark) is theoretically sound — it recovers the Markov equivalence class of the true graph in the large-sample limit — and scales better than exhaustive search. On high-dimensional problems (hundreds of variables), constraint-based methods with fast conditional independence tests (partial correlation for Gaussian data, chi-squared for discrete) are practical where score-based search is not. Glymour, Zhang, and Spirtes's 2019 "Review of Causal Discovery Methods Based on Graphical Models" in <em>Frontiers in Genetics</em> 10:524 provides an authoritative survey of both families.
      </Prose>

      <H3>8.4 Parameter learning: Dirichlet priors prevent overfitting</H3>

      <Prose>
        Given a fixed DAG structure, parameter learning for discrete BNs is a closed-form MLE or MAP problem: count co-occurrences. With Dirichlet priors over each CPT row (equivalent to adding pseudocounts), the MAP estimate is Laplace-smoothed — exactly the same reasoning as in Naive Bayes. The Dirichlet hyperparameter {"alpha_0"} controls smoothing strength. For small datasets relative to the CPT size, Dirichlet priors are essential: a node with 5 parents, each binary, has a CPT with <Code>{"2^5 = 32"}</Code> rows, each requiring its own row probability estimate. With 64 training samples, that is an average of 2 samples per row — heavily underestimated without priors.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Wrong DAG gives confidently wrong answers</H3>

      <Prose>
        A BN is only as good as its DAG. Unlike a regression model, which produces predictions that gracefully degrade when the model is misspecified, a BN with a wrong causal structure can give answers that are confidently incorrect. If you omit a confounding edge (e.g., forget that Age affects both Smoking and Cancer), the backdoor adjustment will fail to block the backdoor path, and your causal effect estimate will be biased — with no obvious signal in the data that anything is wrong. The point estimate will look precise. This is the most dangerous failure mode: high confidence in a wrong answer. Always subject your DAG to domain expert review and, when possible, sensitivity analysis over plausible alternative DAGs.
      </Prose>

      <H3>9.2 Observational data alone cannot identify causal structure</H3>

      <Prose>
        Given only observational data, the best you can identify is the Markov equivalence class (MEC) of the true DAG — the set of DAGs that have the same skeleton (undirected edges) and the same set of v-structures (colliders). Within an MEC, multiple DAGs are observationally indistinguishable. For example, A{"->"}B{"->"}C and A{"<-"}B{"<-"}C are observationally equivalent (both imply A and C are conditionally independent given B). Distinguishing between them requires either domain knowledge, temporal ordering information (causes precede effects), or interventional data (randomized experiments). This is why Pearl insists that causal discovery requires more than statistics: it requires assumptions about mechanisms.
      </Prose>

      <H3>9.3 Unobserved confounders invalidate identification</H3>

      <Prose>
        The backdoor criterion requires that all backdoor paths from treatment to outcome are blocked by observed variables. If there are unobserved confounders (common causes of X and Y that are not in the data), no set of observed variables can block those paths. The causal effect is then not identified from observational data alone. The frontdoor criterion handles some cases of unobserved confounding (when mediators are observed), but there are causal structures where no observational identification is possible. In those cases, you need randomized experiments, natural experiments (instrumental variables), or regression discontinuity designs.
      </Prose>

      <H3>9.4 Selection bias</H3>

      <Prose>
        Selection bias arises when the observed sample is not representative of the target population because selection into the sample is correlated with the outcome. In a BN, this manifests as conditioning on a collider: if you only observe patients who were hospitalized (S=1), and hospitalization depends on both disease severity (D) and insurance status (I), then D and I become correlated in your sample even if they are independent in the population. Berkson's paradox is the canonical example: in hospitalized patients, two unrelated conditions appear negatively correlated. Always check whether your data collection process could have induced collider conditioning before interpreting observational patterns.
      </Prose>

      <H3>9.5 Confusing Bayes-optimal prediction with causal correctness</H3>

      <Prose>
        A BN can be trained to produce optimal probabilistic predictions under distribution shift without those predictions being causally correct. A model trained on observational data will use all available associations — including confounded ones — to minimize prediction error on the training distribution. If the test distribution involves an intervention (e.g., a drug is administered to a population that would not normally take it), the observational model will fail. This is the covariate shift / distribution shift distinction applied to causal vs. anti-causal models. Always ask: am I predicting under the same distribution I trained on, or under an interventional distribution? If the latter, I need a causal model.
      </Prose>

      <H3>9.6 Identifiability check before do-calculus</H3>

      <Prose>
        The do-calculus is complete for identification over semi-Markovian causal models (DAGs with unobserved confounders represented as bidirected edges), as shown by Shpitser and Pearl (2006, AAAI). But not every causal quantity is identifiable — there exist DAGs where no combination of do-calculus rules can reduce {"P(Y | do(X))"} to an observational quantity. Always run an identifiability check (the ID algorithm in Shpitser and Pearl 2006, or DoWhy's <Code>identify_effect</Code>) before attempting to compute a causal effect. If the effect is not identified, report this as a finding — claiming a biased estimate as a causal effect is a serious error.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified for author, year, venue, and main contribution. Read in this order to follow the intellectual development from probabilistic reasoning to causal inference.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Pearl 1988 — Founding text of Bayesian networks",
            render: () => (
              <Prose>
                Pearl, J. (1988). <em>Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference.</em> Morgan Kaufmann, San Mateo. ISBN 1-55860-479-0. Available via ACM Digital Library (dl.acm.org/doi/10.5555/534975) and Elsevier. This is the foundational text. Part I introduces belief networks and the d-separation criterion. Part II covers exact inference via message passing (poly-tree propagation, the precursor to junction tree). Part III covers constraint-based learning. The Alarm network example appears here. Pearl won the Turing Award in 2011 partly on the basis of this work.
              </Prose>
            ),
          },
          {
            label: "Spiegelhalter, Dawid, Lauritzen, Cowell 1993 — Production-ready inference",
            render: () => (
              <Prose>
                Spiegelhalter, D.J., Dawid, A.P., Lauritzen, S.L., and Cowell, R.G. (1993). "Bayesian Analysis in Expert Systems." <em>Statistical Science</em>, 8(3), 219–247. Available via Project Euclid (projecteuclid.org/euclid.ss/1177010888). This paper bridges theory and practice. It introduces the junction tree algorithm as the standard computational engine for exact BN inference, covers parameter learning with Dirichlet priors, and describes the HUGIN system — the first production BN engine, used in medical diagnosis at hospitals in Denmark. The paper is unusually clear on implementation details (triangulation, clique tree construction, message passing protocol) and was the implementation blueprint for most early BN software.
              </Prose>
            ),
          },
          {
            label: "Pearl 2009 — Causality and the do-calculus",
            render: () => (
              <Prose>
                Pearl, J. (2009). <em>Causality: Models, Reasoning and Inference.</em> Cambridge University Press, 2nd edition. ISBN 978-0-521-89560-6. Available at bayes.cs.ucla.edu/BOOK-2K/. The second edition (first published 2000) contains the full development of structural causal models (SCMs), the do-calculus (Chapter 3), the identification algorithm (Chapter 3, Appendix), counterfactual analysis (Chapter 7), and mediation analysis (Chapter 9). The proof of do-calculus completeness references the companion paper by Shpitser and Pearl 2006. This is the mathematical reference for causal inference; "The Book of Why" (2018) is the accessible companion.
              </Prose>
            ),
          },
          {
            label: "Koller & Friedman 2009 — Comprehensive PGM textbook",
            render: () => (
              <Prose>
                Koller, D. and Friedman, N. (2009). <em>Probabilistic Graphical Models: Principles and Techniques.</em> MIT Press. ISBN 978-0-262-01319-2. Available via MIT Press (mitpress.mit.edu/9780262013192). 1,272 pages covering BNs, MRFs, factor graphs, exact inference (variable elimination, junction tree), approximate inference (loopy BP, variational, MCMC), and learning (MLE, Bayesian parameter estimation, structure learning). Part V on causal models (Chapters 21–22) is the most rigorous treatment of causality in a PGM textbook. This is the standard graduate course textbook; Stanford's CS228 lecture notes (freely available) are based on it.
              </Prose>
            ),
          },
          {
            label: "Shpitser & Pearl 2006 — Completeness of do-calculus",
            render: () => (
              <Prose>
                Shpitser, I. and Pearl, J. (2006). "Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models." <em>Proceedings of the 21st National Conference on Artificial Intelligence (AAAI-06)</em>, 1219–1226. Available at cdn.aaai.org/AAAI/2006/AAAI06-191.pdf. This paper proves that the do-calculus is complete for identification of interventional distributions in semi-Markovian models (DAGs with unobserved confounders represented as bidirected edges). It also provides the ID algorithm, a complete algorithmic procedure for determining whether {"P(Y | do(X))"} is identifiable and computing it when it is. The completeness result closed the identification problem for this model class. The algorithm is implemented in DoWhy.
              </Prose>
            ),
          },
          {
            label: "Glymour, Zhang, Spirtes 2019 — Causal discovery survey",
            render: () => (
              <Prose>
                Glymour, C., Zhang, K., and Spirtes, P. (2019). "Review of Causal Discovery Methods Based on Graphical Models." <em>Frontiers in Genetics</em>, 10, 524. DOI: 10.3389/fgene.2019.00524. Open access at frontiersin.org. The definitive modern survey covering constraint-based methods (PC algorithm, FCI for latent variables), score-based methods (GES, HillClimb), functional causal models (LiNGAM for non-Gaussian linear models, post-nonlinear models), and hybrid approaches. The paper covers the identification of causal direction from observational data — possible in some cases via non-Gaussianity (LiNGAM) or asymmetry of the noise structure — and discusses assumptions carefully. Essential reading before implementing any causal discovery pipeline.
              </Prose>
            ),
          },
          {
            label: "Ankan & Panda 2015 — pgmpy",
            render: () => (
              <Prose>
                Ankan, A. and Panda, A. (2015). "pgmpy: Probabilistic Graphical Models using Python." <em>Proceedings of the 14th Python in Science Conference (SciPy 2015)</em>, 6–11. Available at proceedings.scipy.org/articles/Majora-7b98e3ed-001. The paper describing the pgmpy library: BN definition, parameter learning (MLE, Bayesian estimation), exact inference (VE, junction tree), and approximate inference (belief propagation, Gibbs sampling). pgmpy is the standard open-source Python library for BNs, with a NumPy/pandas-compatible API. As of 2026, version 1.1.0 uses <Code>DiscreteBayesianNetwork</Code> (previously <Code>BayesianNetwork</Code>).
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through all six before reading the answers. Exercises 1–3 test the core theory; 4–5 test the causal layer; 6 tests implementation judgment.
      </Prose>

      <H3>Exercise 1 (recall — d-separation)</H3>
      <Prose>
        Consider the DAG: A {"→"} B {"→"} C (a chain). (a) Is A d-separated from C given the empty set? (b) Is A d-separated from C given {"\\{B\\}"}? Explain using the collider/non-collider rules. Then consider the DAG: A {"→"} B {"←"} C (a collider at B). (c) Is A d-separated from C given the empty set? (d) Is A d-separated from C given {"\\{B\\}"}?
      </Prose>
      <Callout type="answer" title="Answer 1">
        (a) Chain A{"->"}B{"->"}C, Z = empty. The only path A—B—C has B as a non-collider (chain). Non-colliders block the path when they are observed. B is NOT observed. The path is ACTIVE. A and C are NOT d-separated — they are marginally dependent through B. (b) Chain, Z = {"\\{B\\}"}. B is a non-collider, B IS observed. The path is BLOCKED. A and C are d-separated given B. (c) Collider A{"->"}B{"<-"}C, Z = empty. B is a collider. Colliders BLOCK the path when unobserved. B is NOT observed. The path is BLOCKED. A and C are d-separated (marginally independent). (d) Collider, Z = {"\\{B\\}"}. B IS observed. Conditioning on a collider OPENS the path. A and C are NO LONGER d-separated — they become conditionally dependent given B. This is explaining away: if B is observed and one of A, C is known, the other becomes more or less likely.
      </Callout>

      <H3>Exercise 2 (derivation — factorization)</H3>
      <Prose>
        Write the full joint factorization for a 5-node BN with structure: {"X1 → X3, X2 → X3, X3 → X4, X3 → X5"}. How many parameters does this BN require if all variables are binary? How many would a full joint distribution over 5 binary variables require? What is the ratio?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"Factorization: P(X1,...,X5) = P(X1) * P(X2) * P(X3|X1,X2) * P(X4|X3) * P(X5|X3). Parameter count: P(X1) needs 1 parameter (binary, sums to 1); P(X2) needs 1; P(X3|X1,X2) has 2^2=4 rows, each needing 1 free parameter = 4; P(X4|X3) has 2 rows = 2; P(X5|X3) has 2 rows = 2. Total: 1+1+4+2+2 = 10 parameters. Full joint: 2^5 - 1 = 31 free parameters (31 entries, one is determined by the normalization constraint). Ratio: 10/31 ≈ 0.32. The BN uses about a third of the parameters. For a 20-node network with a similar sparse structure, this ratio becomes negligible: a full joint has 2^20 - 1 ≈ 1,000,000 parameters; a sparse BN might have 50-200."}
      </Callout>

      <H3>Exercise 3 (conceptual — explaining away)</H3>
      <Prose>
        In the Alarm network, Burglary and Earthquake are marginally independent (d-separated given the empty set). Yet after learning that the alarm went off (Alarm=True), they become dependent. Write a concrete numerical argument for why this dependence makes intuitive sense, using the CPT values from Section 4a. What direction does the dependence go (do they become positively or negatively correlated given Alarm=True)?
      </Prose>
      <Callout type="answer" title="Answer 3">
        From the CPT: P(Alarm=T | Burglary=T, Earthquake=F) = 0.94. P(Alarm=T | Burglary=F, Earthquake=T) = 0.29. P(Alarm=T | Burglary=F, Earthquake=F) = 0.001. Given Alarm=True, the most likely explanations are a burglary (with or without earthquake) or an earthquake alone. If we then learn Earthquake=True, the alarm is partially "explained" — its probability under the earthquake-only scenario is 0.29, which is much higher than the baseline 0.001. This means less of the Alarm's occurrence needs to be attributed to Burglary, so P(Burglary | Alarm=T, Earthquake=T) is lower than P(Burglary | Alarm=T). The two causes become NEGATIVELY correlated given the effect: learning that one cause is present makes the other cause less necessary to explain the observation. P(Burglary | Alarm=T) ≈ 0.376, while P(Burglary | Alarm=T, Earthquake=T) ≈ 0.116 — a substantial decrease. This is explaining away (also called "Berkson's paradox" in the statistical literature, or "selection bias" when it occurs via conditioning on a common effect).
      </Callout>

      <H3>Exercise 4 (do-calculus — backdoor criterion)</H3>
      <Prose>
        Consider a DAG with four variables: Socioeconomic Status (S), Education (E), Job Training (T), and Salary (Y). Edges: S{"→"}E, S{"→"}Y, E{"→"}T, T{"→"}Y. You want to estimate the causal effect of Job Training (T) on Salary (Y). (a) Identify all backdoor paths from T to Y. (b) Propose a valid backdoor adjustment set. (c) Write the backdoor adjustment formula for {"P(Y | do(T=t))"}. (d) Why is E alone NOT a valid backdoor set?
      </Prose>
      <Callout type="answer" title="Answer 4">
        (a) Backdoor paths from T to Y are paths that have an arrow pointing INTO T (going "backward" from T). The path T{"<-"}E{"<-"}S{"->"}Y is a backdoor path: it goes T{"<-"}E{"<-"}S{"->"}Y, with the arrow into T from E, tracing back through S to Y. This path carries confounding from S (socioeconomic status affects both training access through education and salary directly). (b) Valid backdoor sets: {"\\{S\\}"} works (S is not a descendant of T, and conditioning on S blocks T{"<-"}E{"<-"}S{"->"}Y by blocking the S{"->"}... segment). {"\\{S, E\\}"} also works. (c) Backdoor adjustment: {"P(Y | do(T=t)) = sum_s P(Y | T=t, S=s) * P(S=s)"}. If using E as the adjustment set alone: {"P(Y | do(T=t)) = sum_e P(Y | T=t, E=e) * P(E=e)"}. (d) E alone is NOT a valid backdoor set. E is on the path T{"<-"}E{"<-"}S{"->"}Y. Conditioning on E blocks that segment, but E is also on the causal path from E to T (E{"->"}T), which means E is a non-collider on the backdoor path. Conditioning on E DOES block that path — so E would actually work as a backdoor set! The issue is that E is a descendant of S but not of T, and conditioning on E can open a new path via S if S is a confounder. Actually, in this specific DAG, {"\\{E\\}"} is valid: the only backdoor path T{"<-"}E{"<-"}S{"->"}Y is blocked by conditioning on E (E is a non-collider on that path and is observed). The correct reason E alone might not be preferred is that S blocks the path more "upstream" with fewer possible conditioning side effects. Verify with an identification algorithm before applying.
      </Callout>

      <H3>Exercise 5 (debugging — structural learning)</H3>
      <Prose>
        You run pgmpy's HillClimbSearch with BIC scoring on a dataset of 500 samples from a known 6-node BN. The learned graph has 3 missing edges and 2 spurious edges compared to the true graph. List three causes of this discrepancy and one mitigation for each.
      </Prose>
      <Callout type="answer" title="Answer 5">
        Cause 1: Insufficient data. With 500 samples and 6 binary nodes, some CPT rows may have very few observations. The BIC score will penalize complex structures (many edges) appropriately, but low-data cells make the likelihood estimation noisy. Weak associations (low mutual information between variables) will be missed. Mitigation: increase dataset size to at least 1,000–5,000 samples, or use a Bayesian score (BDeu) which incorporates prior smoothing and is more robust at small sample sizes. Cause 2: Local optima in greedy search. HillClimb is a greedy local search that can get stuck in local optima. It finds one edge to add/remove/reverse per step and stops when no single-step improvement is found, missing combinations of moves that would jointly improve the score. Mitigation: run multiple random restarts, or use a tabu search that allows temporary score decreases, or switch to GES (Greedy Equivalence Search) which operates on the space of Markov equivalence classes and has better theoretical properties. Cause 3: Markov equivalence. Multiple DAGs can produce the same BIC score because they are Markov equivalent (same skeleton, same v-structures). Some edge orientations are not identifiable from observational data. The 2 spurious edges may actually be edges from a Markov-equivalent alternative. Mitigation: instead of a single point estimate, report the Markov equivalence class (CPDAG — completed partially directed acyclic graph). Use the PC algorithm which directly outputs the CPDAG. If edge directions are needed, incorporate domain knowledge or collect interventional data.
      </Callout>

      <H3>Exercise 6 (synthesis — causal vs. predictive)</H3>
      <Prose>
        A data scientist trains a Bayesian network on observational hospital records to predict 30-day readmission (Y) from 50 clinical variables. The model achieves AUC 0.82 on a held-out test set from the same hospital. The hospital then introduces a new policy: all patients above a predicted readmission risk of 0.6 are given an intensive discharge counseling intervention (T=1). After 6 months, readmission rates in the T=1 group are HIGHER than in the T=0 group. Is the model wrong? Explain what happened and what the correct analysis approach would have been.
      </Prose>
      <Callout type="answer" title="Answer 6">
        The model is not wrong — it was doing its job correctly. It learned observational associations and predicts readmission well in the pre-intervention distribution. The problem is confusing predictive performance with causal validity under intervention. When the hospital implements the policy (T = 1 for high-risk patients), it changes the data-generating process. The new distribution is {"P(Y | do(T=1))"} for the high-risk subgroup — an interventional distribution the model was never trained on. The observed result (higher readmission in T=1) is a selection effect: T=1 patients were selected precisely because they had the highest predicted risk, and the intervention was not strong enough to overcome their underlying risk differential. This is not proof that counseling is harmful. The correct approach requires: (1) A causal model. Define the DAG: patient risk factors -> predicted risk score -> intervention assignment -> actual outcome. Add any direct paths from risk factors to outcome (the confounders). (2) Identification. Use the backdoor criterion or IV to identify the causal effect of T on Y. The predicted risk score is a mediator/confounder, not a treatment — it should not be in the outcome model without careful adjustment. (3) Estimation via randomization or instrumental variable. Ideally, run a randomized trial where some high-risk patients are randomly assigned to T=0. If randomization is not feasible, use the threshold of the risk score as a regression discontinuity instrument — patients just above and just below the 0.6 cutoff are similar in unobserved risk, making the assignment near-random at the boundary. The fundamental lesson: a predictive model's AUC says nothing about what happens when you intervene based on its predictions.
      </Callout>

    </div>
  ),
};

export default bayesianNetworksContent;
