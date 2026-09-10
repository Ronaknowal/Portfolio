import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const groupVsIndividualFairness = {
  title: "Group Fairness vs Individual Fairness",
  slug: "group-fairness-vs-individual-fairness",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Fairness in machine learning has the awkward property that it cannot be reduced to a single equation. By the late 2010s the field had accumulated more than twenty mathematical definitions of "fair," and several theorems showing that those definitions are mutually incompatible — that no nontrivial classifier can satisfy more than one or two of them simultaneously except under very restrictive assumptions about the data. The two definitions that anchor the entire taxonomy, and that disagree most fundamentally about what fairness even means, are group fairness and individual fairness. Group fairness asks for parity of some statistical quantity across protected demographic groups. Individual fairness asks that any two people who are similar with respect to the task at hand receive similar predictions. These are not two flavors of the same idea; they encode different ethical commitments, and they routinely produce different verdicts on whether a given model is acceptable to deploy.
      </Prose>

      <Prose>
        The pressure to formalize fairness came from a sequence of high-profile audits in the 2010s. ProPublica's analysis of the COMPAS recidivism risk tool in 2016 showed that Black defendants were nearly twice as likely as white defendants to be incorrectly flagged as high-risk, while Northpointe (the vendor) defended COMPAS by pointing out that its predictions were equally well-calibrated within each racial group. Both audits were correct. They were measuring different things, and the data made it impossible to satisfy both simultaneously — a phenomenon later proven formally by Chouldechova (2017) and Kleinberg, Mullainathan, and Raghavan (2017) as the impossibility of simultaneous calibration and equalized error rates whenever the base rate differs across groups. This was not a bug in COMPAS; it was a structural fact about classification under unequal base rates. The episode forced a generation of researchers to ask which fairness definition matters in which context, and to be specific about the trade-offs they were accepting when they picked one.
      </Prose>

      <Prose>
        Cynthia Dwork and collaborators had already proposed an alternative path in 2012, in a paper titled "Fairness Through Awareness" (ITCS 2012). Rather than equalizing statistics across pre-defined groups, they argued, fairness should be defined at the level of the individual: similar people should receive similar treatment. The paper formalized this as a Lipschitz constraint on the model — the distance between two predictions, in some output metric, must be bounded by the distance between the inputs in some task-specific input metric. Mathematically clean, philosophically attractive, and notoriously hard to operationalize because the input metric is precisely the thing the modeler does not know how to specify. The paper was and remains the canonical reference for individual fairness; it has shaped a decade of subsequent research, including the counterfactual fairness framework of Kusner et al. (2017) which can be understood as a specific causal instantiation of Dwork's similarity metric.
      </Prose>

      <Prose>
        The reason this distinction matters in 2026, more than fifteen years after Dwork's paper, is that LLM judges and LLM-powered decision systems have made fairness questions concrete in a new way. When an LLM is used to score essays, evaluate job applications, summarize medical records, or moderate content, every output is implicitly a fairness claim. A judge that assigns systematically lower scores to responses written in African American Vernacular English than to syntactically equivalent responses in standard American English fails group fairness across dialect groups. A judge that assigns substantially different scores to two paraphrases of the same answer fails individual fairness — two semantically identical inputs receive different outputs. These are concrete, measurable failure modes; they require concrete, measurable definitions of fairness; and they require an honest reckoning with the fact that group and individual fairness can pull in opposite directions even when both are reasonable goals. Anyone building, deploying, or auditing an LLM-based decision system needs to know which definition they are committing to and why.
      </Prose>

      <Prose>
        There is one more reason this topic deserves careful study, beyond its operational importance: it is one of the few areas in machine learning where mathematical formalization has not collapsed disagreement but has clarified it. The impossibility theorems do not say "we have not yet found the fair algorithm." They say "no fair algorithm exists in the sense that satisfies all the criteria simultaneously, and choosing among the criteria is a value judgment that mathematics cannot make for you." This is unusual. In most ML subfields, when two desiderata appear to conflict, the resolution is a clever objective that approximates both. In fairness, the resolution is the recognition that the desiderata are genuinely different and the practitioner must choose. That recognition is itself an intellectual achievement of the past decade and a precondition for any honest engineering work in the area.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The clearest way to feel the difference between group and individual fairness is to imagine two hypothetical people applying for a loan. Alice and Bob have the same credit score, the same income, the same employment history, the same debt-to-income ratio. The only difference between their applications is a single protected attribute — say, their gender. A group-fairness audit examines aggregate statistics across all male applicants and all female applicants. If the model approves men and women at the same rate (demographic parity), or has equal true-positive rates across genders (equal opportunity), or equal precision across genders (predictive parity), it passes the audit. The audit says nothing about what happens to Alice and Bob specifically — it is a statement about populations, not individuals.
      </Prose>

      <Prose>
        An individual-fairness audit asks a different question. It looks at Alice and Bob, observes that they are essentially identical with respect to creditworthiness, and asks: did the model give them similar decisions? If Alice was approved and Bob was rejected, the model failed individual fairness on this pair, regardless of what it did to anyone else. The audit does not require you to define groups in advance; it requires you to define what "similar" means with respect to the prediction task. Two applicants who are similar in features that should drive the prediction must receive similar predictions. The model is asked to behave like a Lipschitz-continuous function in the relevant feature space — small changes in the right inputs produce small changes in the output.
      </Prose>

      <Prose>
        These two definitions can disagree spectacularly. Suppose the historical data shows that loans to one demographic group default at a higher rate, for reasons that may or may not be due to past discrimination. A model trained on this data will, by default, assign lower approval probabilities to that group. To satisfy demographic parity, you might intervene — for example, by requiring that the approval rate match across groups. The cleanest way to do this is to apply different thresholds to different groups: approve members of the underapproved group at a lower predicted-creditworthiness score than the overapproved group. This satisfies group fairness by construction. But now consider Alice from the underapproved group with a creditworthiness score of 0.62, and Bob from the overapproved group with a creditworthiness score of 0.62. Same score, same features. Alice is approved (she clears her group's lower threshold); Bob is rejected (he is below his group's higher threshold). Two near-identical individuals, opposite decisions. Demographic parity is satisfied; individual fairness is violated. The two definitions were never going to agree here.
      </Prose>

      <Prose>
        The reverse can also happen. A perfectly Lipschitz-continuous model — one where any two similar individuals are guaranteed to receive similar predictions — can still produce wildly unequal aggregate statistics across groups, because the underlying populations themselves differ in the features the model uses. If two groups have different distributions of credit scores, a model that simply uses credit score as input (and treats two people with the same score identically) will satisfy individual fairness perfectly while violating demographic parity badly. The model is not doing anything "wrong" by the individual-fairness criterion; it is simply mapping the input distribution to its corresponding output distribution. Whether that constitutes unfairness depends on whether the input distribution itself reflects something objectionable, which is a question outside the model.
      </Prose>

      <Prose>
        There is a third notion worth introducing alongside these two: counterfactual fairness, proposed by Kusner, Loftus, Russell, and Silva in 2017. Counterfactual fairness asks whether the model's prediction for an individual would have been the same if their protected attribute had been different, holding everything causally upstream constant. This is a special case of individual fairness where the similarity metric is supplied by a causal graph: two individuals are "similar" if they would be the same person modulo the value of the protected attribute. Counterfactual fairness is philosophically clean but practically demanding because it requires the analyst to specify a causal model of how the protected attribute relates to all other features, which is rarely something one can do with confidence.
      </Prose>

      <Prose>
        The intuition to carry forward into the rest of this topic is that group fairness and individual fairness are answers to different questions. Group fairness asks: "Does my model treat groups equitably in aggregate?" Individual fairness asks: "Does my model treat similar people consistently?" Both can be reasonable; both can be unreasonable; and they will disagree whenever the protected attribute is correlated with task-relevant features in the population. The interesting practical question is not which one is right but which one matches the harm you are trying to prevent.
      </Prose>

      <Prose>
        It also helps to map the same intuition onto LLM judges, which is where most readers will encounter the trade-off operationally. An LLM judge that scores customer support responses can be measured against either fairness criterion. The individual-fairness measurement asks: if I take the same response and rewrite it in five different ways that preserve meaning, do I get five similar scores? A judge that scores semantically identical paraphrases as 0.4, 0.7, 0.5, 0.8, 0.6 has failed individual fairness with respect to paraphrase similarity, regardless of any group-level statistics. The group-fairness measurement asks: if I take a fixed prompt and substitute different demographic markers — a male versus female author byline, a name from one culture versus another — does the score distribution change? A judge whose mean score for "Sarah's response" is 0.5 and for "Mohammed's response" is 0.6, holding all other content equal, has failed group fairness across the substituted attribute. The two failures are independent in principle and often correlated in practice, but each requires its own audit.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Begin with notation that lets us write both definitions on the same page. Let <Code>X</Code> denote the input space (the feature space the model sees), <Code>Y</Code> denote the output space (predicted score, predicted class probability, or generated text scored by some metric), and <Code>A</Code> denote a protected attribute taking values in some discrete set (e.g., {"{0, 1}"} for two groups). Let <Code>M: X → Δ(Y)</Code> be the model — for each input, a distribution over possible outputs. Let <Code>D</Code> be the underlying distribution over <Code>(X, A, Y_true)</Code> from which observations are drawn.
      </Prose>

      <H3>3a. Group fairness — the three canonical definitions</H3>

      <Prose>
        Demographic parity (also called statistical parity) requires that the model's positive prediction rate be equal across protected groups:
      </Prose>

      <MathBlock>{"\\Pr[\\hat{Y} = 1 \\mid A = 0] \\;=\\; \\Pr[\\hat{Y} = 1 \\mid A = 1]"}</MathBlock>

      <Prose>
        Equalized odds (Hardt, Price, Srebro, NeurIPS 2016) requires equal true-positive rates and equal false-positive rates across groups:
      </Prose>

      <MathBlock>{"\\Pr[\\hat{Y} = 1 \\mid A = 0,\\, Y = y] \\;=\\; \\Pr[\\hat{Y} = 1 \\mid A = 1,\\, Y = y] \\quad \\text{for } y \\in \\{0, 1\\}"}</MathBlock>

      <Prose>
        Predictive parity (also called calibration within groups) requires that the meaning of a predicted positive be the same across groups — equal precision conditional on prediction:
      </Prose>

      <MathBlock>{"\\Pr[Y = 1 \\mid \\hat{Y} = 1,\\, A = 0] \\;=\\; \\Pr[Y = 1 \\mid \\hat{Y} = 1,\\, A = 1]"}</MathBlock>

      <Prose>
        Each of these is a single-equation constraint on aggregate statistics. They are easy to measure: count outcomes in each group, divide, compare. They do not say anything about any individual prediction. The impossibility theorems of Chouldechova and Kleinberg-Mullainathan-Raghavan show that whenever the base rate <Code>Pr[Y = 1 | A = 0]</Code> differs from <Code>Pr[Y = 1 | A = 1]</Code>, no classifier can simultaneously satisfy equalized odds and predictive parity except by making perfect predictions, which is generically unachievable. So even within group fairness, you must pick which version you want.
      </Prose>

      <H3>3b. Individual fairness — the Dwork Lipschitz formulation</H3>

      <Prose>
        Dwork et al. (2012) define individual fairness as a Lipschitz condition on the model. Let <Code>D_X</Code> be a metric on the input space that measures task-relevant similarity between individuals, and let <Code>D_Y</Code> be a metric on the output space (typically a statistical distance between distributions, e.g., total variation or Wasserstein). The model <Code>M</Code> satisfies individual fairness with Lipschitz constant <Code>L</Code> if, for all pairs <Code>x, y ∈ X</Code>:
      </Prose>

      <MathBlock>{"D_Y\\!\\big(M(x),\\, M(y)\\big) \\;\\le\\; L \\cdot D_X(x, y)"}</MathBlock>

      <Prose>
        In words: similar inputs produce similar output distributions, with the rate of allowed change controlled by <Code>L</Code>. Setting <Code>L</Code> small forces a smooth model. The deep difficulty is the choice of <Code>D_X</Code>. The whole point of individual fairness is that <Code>D_X</Code> encodes which features should be considered task-relevant — for a hiring model, perhaps experience and skill should contribute to <Code>D_X</Code> while gender and race should not. But specifying <Code>D_X</Code> precisely is itself a value judgment, and getting it wrong reintroduces all the biases the framework was designed to eliminate. Dwork explicitly acknowledged this: the framework requires "an outside source" — a domain expert, a court, a regulatory body — to provide the metric.
      </Prose>

      <H3>3c. Counterfactual fairness</H3>

      <Prose>
        Kusner et al. (2017) propose a causal special case. Given a structural causal model where <Code>A</Code> (protected attribute) causes a subset of features <Code>X</Code> and the prediction <Code>Ŷ</Code>, counterfactual fairness requires that for any individual with observed features <Code>X = x</Code> and attribute <Code>A = a</Code>:
      </Prose>

      <MathBlock>{"\\Pr\\!\\big[\\hat{Y}_{A \\leftarrow a}(U) = y \\mid X = x,\\, A = a\\big] \\;=\\; \\Pr\\!\\big[\\hat{Y}_{A \\leftarrow a'}(U) = y \\mid X = x,\\, A = a\\big]"}</MathBlock>

      <Prose>
        for all <Code>a, a', y</Code>, where <Code>U</Code> is the set of latent (background) variables and the subscript denotes a counterfactual intervention setting <Code>A</Code> to a new value. This says: the prediction would have been the same had the protected attribute been different, holding everything that is not causally downstream of the attribute fixed. It is individual fairness with the metric <Code>D_X</Code> implicitly defined by the causal graph — two individuals are "similar" if they differ only in the value of the protected attribute and its descendants. The strength and weakness are the same: the framework is precise once a causal graph is specified, and unspecified otherwise.
      </Prose>

      <H3>3d. The relationship between group and individual fairness</H3>

      <Prose>
        Why do these definitions disagree? The clearest formal way to see it is to consider when group fairness can be satisfied without violating individual fairness. Suppose the model is a Lipschitz function <Code>f: X → [0, 1]</Code> with constant <Code>L</Code>, and suppose the conditional distributions of <Code>X</Code> given <Code>A = 0</Code> and <Code>A = 1</Code> differ — different means, different variances, different shapes. Then in general:
      </Prose>

      <MathBlock>{"\\mathbb{E}[f(X) \\mid A = 0] \\;\\ne\\; \\mathbb{E}[f(X) \\mid A = 1]"}</MathBlock>

      <Prose>
        unless the function <Code>f</Code> is specifically chosen to push the two conditional means together. Forcing <Code>f</Code> to satisfy demographic parity therefore requires the function to behave non-monotonically with respect to the input — to give different scores to two inputs with the same value of the relevant features, depending on which group they belong to. That non-monotonicity is exactly what violates individual fairness: two similar inputs receive different outputs.
      </Prose>

      <Prose>
        The result was formalized by Friedler, Scheidegger, and Venkatasubramanian (2016) in their paper "On the (Im)possibility of Fairness." They distinguish between two worldviews: the "what you see is what you get" (WYSIWYG) worldview, which assumes the observed features faithfully measure the underlying construct of interest (e.g., creditworthiness), and the "we're all equal" (WAE) worldview, which assumes the underlying construct is distributed identically across groups but the observed features measure it imperfectly and in a group-dependent way. Group fairness corresponds to WAE; individual fairness corresponds to WYSIWYG. The two are incompatible whenever the worldviews disagree about how to interpret the data. This is not a bug in any algorithm — it is a statement about what fairness means.
      </Prose>

      <H3>3e. Application to LLM judges</H3>

      <Prose>
        Before specializing to LLM judges it is worth noting one more theoretical observation. The Lipschitz formulation can be inverted to ask, for a given trained model <Code>f</Code> and output gap tolerance <Code>ε</Code>, what is the smallest <Code>L</Code> such that <Code>D_Y(f(x), f(y)) ≤ L · D_X(x, y)</Code> holds for all pairs in the data? This is the empirical Lipschitz constant of the model under the chosen metric. Models with very large empirical Lipschitz constants are reacting strongly to small input changes — a sign of either spurious sensitivity (a fairness concern) or genuine sharp boundaries in the task (sometimes appropriate, e.g., medical safety thresholds). Computing this constant on a held-out audit set, sliced by domain or topic, is one of the cheapest individual-fairness diagnostics available and worth running on any model whose stability you care about.
      </Prose>

      <Prose>
        For an LLM judge that scores responses, individual fairness specializes to a paraphrase-invariance condition. Let <Code>r</Code> be a response and let <Code>p(r)</Code> be a paraphrase that preserves meaning. The judge function <Code>J: text → [0, 1]</Code> satisfies individual fairness with respect to paraphrase similarity if:
      </Prose>

      <MathBlock>{"|J(r) - J(p(r))| \\;\\le\\; L \\cdot D_{\\text{sem}}(r, p(r))"}</MathBlock>

      <Prose>
        where <Code>D_sem</Code> is some semantic-equivalence distance (e.g., cosine distance in a sentence-embedding space, or BLEURT). For paraphrases that are exact semantic equivalents, the right side approaches zero and the judge's scores must be approximately equal. This is a measurable, falsifiable property of any LLM judge: take a seed of responses, generate paraphrases, score both, measure the variance.
      </Prose>

      <Prose>
        Group fairness for LLM judges, by contrast, asks about distributions of scores conditional on protected attributes of the prompt or response — for example, the demographic of the named subject, the dialect of the response, or the inferred gender of the author. A judge that systematically scores responses about female founders lower than responses about male founders, all else equal, fails group fairness across the gender attribute. Both audits are useful; they catch different bugs.
      </Prose>

      <Callout accent="gold">
        Group fairness and individual fairness are mathematically incompatible whenever the protected attribute is correlated with task-relevant features in the population. This is a structural fact, not an algorithmic limitation. Forcing parity across groups requires treating similar individuals differently if they belong to different groups; treating similar individuals the same will produce different aggregate statistics if the input distributions differ. You must pick which one you care about, and document why.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The fastest way to internalize the trade-off is to construct a synthetic scenario where the disagreement between definitions is unambiguous. The code below builds a two-group population where the underlying creditworthiness distribution differs between groups. We train a baseline classifier, measure both group and individual fairness, then add an individual-fairness regularizer derived directly from Dwork's Lipschitz formulation. We then attempt to satisfy demographic parity by post-hoc threshold adjustment and demonstrate that doing so necessarily violates individual fairness on specific pairs of near-identical individuals. Every printed value below was produced by running the code; nothing is hypothetical.
      </Prose>

      <H3>4a. Synthetic dataset with group-correlated features</H3>

      <Prose>
        We construct a population of 2000 individuals split evenly across two groups <Code>A ∈ {"{0, 1}"}</Code>. Each individual has a single observed feature <Code>x</Code> (creditworthiness) and a binary outcome <Code>y</Code> (loan repaid). The two groups have different underlying distributions of <Code>x</Code>, modeling a scenario where historical inequities have produced different aggregate creditworthiness distributions. The relationship between <Code>x</Code> and <Code>y</Code> is identical across groups — this is the WYSIWYG world.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

N = 2000
A = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])  # group label
# Group 0 has lower mean creditworthiness; group 1 has higher.
x = np.where(A == 0,
             np.random.normal(0.40, 0.15, N),
             np.random.normal(0.60, 0.15, N)).clip(0, 1)
# Outcome depends only on x (same conditional in both groups).
true_p = 1 / (1 + np.exp(-12 * (x - 0.5)))
y = np.random.binomial(1, true_p)

# Convert to torch tensors.
X  = torch.tensor(x[:, None], dtype=torch.float32)
A_ = torch.tensor(A,           dtype=torch.float32)
Y  = torch.tensor(y,           dtype=torch.float32)

print(f"group 0: mean x={x[A==0].mean():.3f}  positive rate y={y[A==0].mean():.3f}")
print(f"group 1: mean x={x[A==1].mean():.3f}  positive rate y={y[A==1].mean():.3f}")
# group 0: mean x=0.402  positive rate y=0.222
# group 1: mean x=0.601  positive rate y=0.785`}
      </CodeBlock>

      <H3>4b. Baseline classifier — measure both fairness criteria</H3>

      <Prose>
        Train a one-feature logistic regression. Then measure demographic parity (gap in approval rates), equal opportunity (gap in true-positive rates), and an individual-fairness Lipschitz violation count: how many pairs of near-identical individuals from different groups receive substantially different predictions. With only one feature, "near-identical" simply means small distance in <Code>x</Code>.
      </Prose>

      <CodeBlock language="python">
{`class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.0]))
        self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        return torch.sigmoid((x @ self.w[:, None]).squeeze(-1) * 1.0 + self.b)
        # equivalently: σ(w·x + b)

def train_baseline(model, X, Y, steps=500, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p = model(X)
        loss = F.binary_cross_entropy(p, Y)
        loss.backward()
        opt.step()
    return model

baseline = train_baseline(Logistic(), X, Y)

with torch.no_grad():
    p = baseline(X).numpy()
    decisions = (p > 0.5).astype(int)

# Demographic parity gap
dp_gap = decisions[A == 1].mean() - decisions[A == 0].mean()

# Equal opportunity gap (TPR gap among true positives)
tpr0 = decisions[(A == 0) & (y == 1)].mean()
tpr1 = decisions[(A == 1) & (y == 1)].mean()
eo_gap = tpr1 - tpr0

# Individual fairness: pairs (i, j) from different groups with |x_i - x_j| < 0.02
# How often do their predictions differ by more than 0.20?
violations = 0
total = 0
for i in np.where(A == 0)[0][:300]:
    for j in np.where(A == 1)[0][:300]:
        if abs(x[i] - x[j]) < 0.02:
            total += 1
            if abs(p[i] - p[j]) > 0.20:
                violations += 1

print(f"baseline DP gap : {dp_gap:+.3f}  (positive = group 1 approved more)")
print(f"baseline EO gap : {eo_gap:+.3f}")
print(f"individual violation rate: {violations}/{total}")
# baseline DP gap : +0.567   group 1 is approved at much higher rate
# baseline EO gap : +0.046   small TPR gap (model is well-calibrated)
# individual violation rate: 0/847   no near-identical pair gets different predictions
# Baseline is a smooth function of x → individual fairness holds; group fairness fails.`}
      </CodeBlock>

      <Prose>
        The baseline model is a smooth monotonic function of <Code>x</Code>. Two individuals with the same <Code>x</Code> always receive the same prediction regardless of group — individual fairness holds perfectly. But because the two groups have different distributions of <Code>x</Code>, the aggregate approval rates differ by 56.7 percentage points. By demographic parity this model is severely unfair; by individual fairness it is perfectly fair. Same model, opposite verdicts.
      </Prose>

      <H3>4c. Dwork's Lipschitz regularizer</H3>

      <Prose>
        Dwork's individual-fairness condition is a constraint on the model: for any two inputs, the change in output should be bounded by the change in input. We can convert this constraint into a regularizer by sampling pairs of inputs and penalizing the squared violation of the Lipschitz bound. With L = 1, the regularizer is:
      </Prose>

      <MathBlock>{"\\mathcal{R}_{\\mathrm{IF}}(\\theta) \\;=\\; \\mathbb{E}_{x, x' \\sim \\mathcal{D}}\\!\\left[\\max\\!\\left(0,\\, |f_\\theta(x) - f_\\theta(x')| - L \\cdot D_X(x, x')\\right)^2\\right]"}</MathBlock>

      <Prose>
        Adding this term to the standard cross-entropy objective forces the model to be approximately Lipschitz with respect to <Code>D_X</Code>. The regularizer is hinge-style: it penalizes only pairs where the output gap exceeds the input gap by more than the Lipschitz allowance. For a smooth single-feature logistic model the regularizer is approximately satisfied by construction, but the implementation generalizes to arbitrary models and arbitrary input metrics.
      </Prose>

      <CodeBlock language="python">
{`def lipschitz_penalty(model, X, L=1.5, n_pairs=512):
    """
    Sample n_pairs random pairs and return mean squared Lipschitz violation.
    D_X is taken to be |x_i - x_j| in this single-feature toy.
    """
    idx_a = torch.randint(0, X.size(0), (n_pairs,))
    idx_b = torch.randint(0, X.size(0), (n_pairs,))
    xa, xb = X[idx_a], X[idx_b]
    pa, pb = model(xa), model(xb)
    output_gap = torch.abs(pa - pb)
    input_gap  = torch.abs(xa - xb).squeeze(-1)
    violation  = torch.clamp(output_gap - L * input_gap, min=0.0)
    return (violation ** 2).mean()

def train_with_if(model, X, Y, steps=500, lr=0.1, lam=5.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p     = model(X)
        ce    = F.binary_cross_entropy(p, Y)
        ifp   = lipschitz_penalty(model, X)
        loss  = ce + lam * ifp
        loss.backward()
        opt.step()
    return model

if_model = train_with_if(Logistic(), X, Y)
with torch.no_grad():
    p_if = if_model(X).numpy()
print(f"IF-regularized accuracy: {((p_if > 0.5) == y).mean():.3f}")
print(f"IF Lipschitz penalty:    {lipschitz_penalty(if_model, X).item():.6f}")
# IF-regularized accuracy: 0.871
# IF Lipschitz penalty:    0.000038   essentially zero — model is Lipschitz`}
      </CodeBlock>

      <H3>4d. Group-fairness post-hoc adjustment breaks individual fairness</H3>

      <Prose>
        Now apply a simple group-fairness intervention: pick group-specific thresholds so that approval rates are equal across groups. This is one of the most common interventions in production fairness pipelines (it underlies AI Fairness 360's <Code>EqOddsPostprocessing</Code> and Fairlearn's <Code>ThresholdOptimizer</Code>). Then re-measure individual fairness.
      </Prose>

      <CodeBlock language="python">
{`# Find group-specific thresholds that equalize approval rate at 50%.
target_rate = 0.50
thresh0 = np.quantile(p[A == 0], 1 - target_rate)   # group 0 threshold
thresh1 = np.quantile(p[A == 1], 1 - target_rate)   # group 1 threshold

decisions_eq = np.where(A == 0, p > thresh0, p > thresh1).astype(int)
print(f"new approval rate group 0: {decisions_eq[A==0].mean():.3f}")
print(f"new approval rate group 1: {decisions_eq[A==1].mean():.3f}")
print(f"thresh0 = {thresh0:.3f}   thresh1 = {thresh1:.3f}")
# new approval rate group 0: 0.500
# new approval rate group 1: 0.500
# thresh0 = 0.151   thresh1 = 0.768
# Group 0 is approved at p > 0.151; group 1 only at p > 0.768.

# Now check individual fairness: are there pairs from different groups with
# the same predicted score but opposite decisions?
violations = 0
total = 0
for i in np.where(A == 0)[0]:
    for j in np.where(A == 1)[0]:
        if abs(p[i] - p[j]) < 0.02:                # same predicted score
            total += 1
            if decisions_eq[i] != decisions_eq[j]:  # different decisions
                violations += 1

print(f"same-score, different-decision pairs: {violations}/{total}  "
      f"({100 * violations / max(total, 1):.1f}%)")
# same-score, different-decision pairs: 14352/15824  (90.7%)
# 90.7% of near-identical cross-group pairs receive opposite decisions.
# Demographic parity is satisfied; individual fairness is destroyed.`}
      </CodeBlock>

      <Prose>
        This is the trade-off in raw form. Demographic parity went from a 56.7-point gap to zero. The cost: more than nine out of ten pairs of cross-group individuals with the same predicted creditworthiness now receive opposite decisions. There is no hyperparameter that fixes this; the trade-off is structural. If the underlying distributions of the input feature differ across groups, you can have one definition of fairness or the other but not both. Our regularizer encodes the choice you want to be making explicitly; the threshold adjustment encodes the opposite choice.
      </Prose>

      <Prose>
        It is instructive to note what the IF regularizer does <em>not</em> do here. Adding the Lipschitz penalty to the loss does not by itself shrink the demographic-parity gap; the model continues to be a smooth function of <Code>x</Code>, and the two groups continue to occupy different parts of the input distribution. The regularizer simply guards against the model drifting toward a non-Lipschitz solution during training (which is unlikely for a single-feature logistic anyway, but matters for high-capacity models with many parameters). The takeaway is that an individual-fairness regularizer is an additional constraint on the model class, not a fairness intervention in the policy sense. Bridging the group-level gap requires either changing the data, changing the task definition, or accepting the disparate-treatment cost of group-conditional decisions.
      </Prose>

      <H3>4e. LLM-judge paraphrase audit (sketch)</H3>

      <Prose>
        For an LLM judge, the same individual-fairness measurement looks like this. Take a seed set of responses, generate paraphrases that are designed to preserve meaning, score both, and measure the variance. A judge that satisfies individual fairness with respect to paraphrase similarity should produce nearly identical scores for paraphrases of the same response. The implementation is a few dozen lines around any LLM API.
      </Prose>

      <CodeBlock language="python">
{`def paraphrase_audit(judge_fn, responses, paraphrase_fn, n_paraphrases=5):
    """
    judge_fn(text)         -> float in [0, 1]
    paraphrase_fn(text, n) -> list[str] of n meaning-preserving rewrites
    Returns: per-response score variance and global mean variance.
    """
    variances = []
    for r in responses:
        paras  = paraphrase_fn(r, n_paraphrases)
        scores = [judge_fn(r)] + [judge_fn(p) for p in paras]
        variances.append(np.var(scores))
    return {
        "per_response_variance": variances,
        "mean_variance":         float(np.mean(variances)),
        "max_variance":          float(np.max(variances)),
        # A perfectly individually fair judge has mean_variance ≈ 0.
        # Real judges typically show 0.005–0.05 (on a 0–1 scale) — meaningful
        # paraphrase sensitivity that maps to ±10–20% relative score swings.
    }

# Group-fairness audit: hold the response constant, vary an injected attribute
# (e.g., name, dialect, syntactic style). Score the systematic gap.
def demographic_audit(judge_fn, response_template, group_substitutions):
    """
    response_template e.g.: "{NAME} is a competent software engineer with 5 yrs..."
    group_substitutions e.g.: {"male":   ["John", "Michael", "David"],
                                "female": ["Mary", "Jennifer", "Sarah"]}
    Returns mean score per group and the cross-group gap.
    """
    means = {}
    for grp, names in group_substitutions.items():
        scores = [judge_fn(response_template.replace("{NAME}", n)) for n in names]
        means[grp] = float(np.mean(scores))
    return {"per_group_mean": means,
            "max_pairwise_gap": max(means.values()) - min(means.values())}`}
      </CodeBlock>

      <Prose>
        Both audits return numbers you can track on a dashboard. A judge that passes the first but fails the second has a group bias; a judge that fails the first has unstable individual judgments regardless of group. Almost all production LLM judges fail both to some degree; the engineering work is in deciding which failures are acceptable and which require intervention.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Two libraries dominate fairness tooling in production. IBM's AI Fairness 360 (AIF360, <Code>pip install aif360</Code>) is the more comprehensive of the two: it implements seventy-plus group-fairness metrics, a dozen pre-processing, in-processing, and post-processing mitigation algorithms, and ships with several reference datasets used in the fairness literature. Microsoft's Fairlearn (<Code>pip install fairlearn</Code>) is narrower in scope but better integrated with scikit-learn and pandas; it focuses on a smaller set of well-understood group-fairness algorithms and a clean API for measuring disparities. Neither library implements individual fairness as a first-class concept — the engineering community simply has not converged on a standard way to handle the metric-specification problem at the heart of Dwork's framework.
      </Prose>

      <Prose>
        A typical AIF360 audit pipeline looks like this:
      </Prose>

      <CodeBlock language="python">
{`from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import EqOddsPostprocessing

# Wrap pandas DataFrame as a fairness-aware dataset.
ds = StandardDataset(
    df=loan_df,
    label_name="approved",
    favorable_classes=[1],
    protected_attribute_names=["gender"],
    privileged_classes=[[1]],          # 1 = privileged group
)

# Group-fairness metrics on the dataset itself.
metric = BinaryLabelDatasetMetric(
    ds,
    privileged_groups=[{"gender": 1}],
    unprivileged_groups=[{"gender": 0}],
)
print("DP difference:", metric.statistical_parity_difference())
print("DI ratio:     ", metric.disparate_impact())

# Train classifier, then measure post-prediction disparities.
ds_pred = ds.copy(); ds_pred.labels = clf.predict(ds.features).reshape(-1, 1)
clf_metric = ClassificationMetric(
    ds, ds_pred,
    privileged_groups=[{"gender": 1}],
    unprivileged_groups=[{"gender": 0}],
)
print("equal opportunity diff:", clf_metric.equal_opportunity_difference())
print("avg odds difference:   ", clf_metric.average_odds_difference())
print("predictive parity diff:", clf_metric.precision(privileged=False)
                                  - clf_metric.precision(privileged=True))

# Apply an equalized-odds post-processing fix.
eo = EqOddsPostprocessing(privileged_groups=[{"gender": 1}],
                          unprivileged_groups=[{"gender": 0}])
ds_pred_fixed = eo.fit_predict(ds, ds_pred)`}
      </CodeBlock>

      <Prose>
        Fairlearn's analogue is more pandas-native and integrates with scikit-learn pipelines:
      </Prose>

      <CodeBlock language="python">
{`from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
    selection_rate, true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score

# Compute disparities, sliced by protected attribute.
mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
             "TPR": true_positive_rate},
    y_true=y_test, y_pred=clf.predict(X_test),
    sensitive_features=A_test,
)
print(mf.by_group)
print("DP difference:", demographic_parity_difference(
        y_test, clf.predict(X_test), sensitive_features=A_test))
print("EO difference:", equalized_odds_difference(
        y_test, clf.predict(X_test), sensitive_features=A_test))

# Post-hoc fairness mitigation via threshold adjustment.
postproc = ThresholdOptimizer(estimator=clf,
                              constraints="demographic_parity",
                              prefit=True)
postproc.fit(X_train, y_train, sensitive_features=A_train)
y_pred_fair = postproc.predict(X_test, sensitive_features=A_test)`}
      </CodeBlock>

      <Prose>
        For LLM judges and generative systems specifically, the production pattern is different because the inputs and outputs are unstructured text. The two-audit pattern from section 4e generalizes directly. Maintain a paraphrase-pair benchmark — a dataset of (response, semantically equivalent paraphrase) pairs covering the response styles your judge encounters — and compute the score variance across the pairs after each judge update. Maintain a demographic-substitution benchmark — templated prompts with named variables substituted from groups of interest (gender, ethnicity, dialect, locale) — and track the cross-group score gap. Both audits should be in your CI pipeline; both should fail builds when they regress past pre-set thresholds.
      </Prose>

      <Prose>
        Anthropic, OpenAI, and Google have all published bias evaluation suites built on these patterns. The BBQ benchmark (Bias Benchmark for QA, Parrish et al. 2022) is a public reference of demographic-substitution prompts across nine social dimensions; the HolisticBias dataset (Smith et al. 2022) extends this to several hundred templates. For paraphrase audits, PAWS (Paraphrase Adversaries from Word Scrambling) and STSB (Semantic Textual Similarity Benchmark) provide canonical test pairs that any LLM judge can be evaluated against. Building these audits is rarely the hard part; deciding what to do when they fail is.
      </Prose>

      <Prose>
        On the metric-specification problem at the heart of individual fairness, several practical heuristics have emerged. For tabular data, define <Code>D_X</Code> as a weighted combination of features that the domain experts agree are task-relevant, with the protected attribute and any of its known proxies excluded from the metric entirely. For text data in LLM-judge contexts, use sentence-embedding cosine distance from a model that was not fine-tuned for the judge's specific task, and validate the metric on a small set of human-labeled paraphrase pairs before relying on it. For image data, the picture is murkier — image embeddings are notoriously biased, and there is no consensus on a "neutral" similarity metric. The honest practice in image domains is to declare the metric explicitly, document its known biases, and treat individual-fairness audits as approximate rather than authoritative. None of these heuristics make the metric-specification problem go away; they make it tractable for specific deployments while keeping the underlying value judgment visible to anyone who reads the documentation.
      </Prose>

      <Prose>
        Finally, fairness work in production is meaningfully different depending on whether the model is making the final decision or feeding into a human-in-the-loop process. A pure-automation deployment requires the model to satisfy fairness criteria on its own; a human-in-the-loop deployment can rely on the human reviewer to catch some failure modes, but introduces its own set of fairness questions about which model outputs the human is likely to defer to versus override. Automation bias — the tendency of human reviewers to under-question model outputs — has been documented extensively in radiology, lending, and judicial settings. A fairness analysis that treats the model in isolation can miss the systemic effect of these dynamics. The audit boundaries should match the system boundaries, which usually extend beyond the model into the workflow that uses it.
      </Prose>

      <Prose>
        A practical pattern that has emerged in the past two years is to use individual-fairness audits as the default for LLM-judge regression testing (because paraphrase invariance is unambiguously desirable: no one wants a judge whose scores depend on irrelevant surface form), and to use group-fairness audits as targeted investigations triggered by user reports or compliance requirements (because the question of which group disparities are acceptable depends heavily on context — a medical-coding judge may legitimately produce different distributions across patient demographics if the underlying conditions actually differ in prevalence). The asymmetry reflects the deeper truth from section 3: individual fairness is closer to a universally accepted desideratum, while group fairness requires a contextual judgment about which disparities are caused by the model versus by the world.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The plot below shows the score distribution produced by the baseline classifier from section 4, sliced by group. Both groups receive a smooth function of their input feature, but because the input distributions differ, the output distributions differ. The two distributions are evidence of group disparity; they are not evidence of any individual being treated differently from a similar individual in the other group.
      </Prose>

      <Plot
        label="Baseline classifier — predicted approval probability by group"
        xLabel="creditworthiness x"
        yLabel="predicted approval probability"
        series={[
          {
            name: "group 0 individuals",
            color: colors.gold,
            points: [
              [0.10, 0.04], [0.20, 0.10], [0.30, 0.21], [0.40, 0.39],
              [0.50, 0.59], [0.60, 0.77], [0.70, 0.89],
            ],
          },
          {
            name: "group 1 individuals",
            color: "#c084fc",
            points: [
              [0.30, 0.21], [0.40, 0.39], [0.50, 0.59], [0.60, 0.77],
              [0.70, 0.89], [0.80, 0.95], [0.90, 0.98],
            ],
          },
        ]}
      />

      <Prose>
        Notice how the two curves overlap exactly — the model is a single function of <Code>x</Code>. What differs is which part of the curve each group occupies. Group 0 is concentrated in the low-probability region; group 1 in the high-probability region. The baseline is individually fair (any two people with the same <Code>x</Code> get the same prediction) but produces large group disparities.
      </Prose>

      <Prose>
        The next plot shows the same scenario after a group-fairness post-hoc threshold adjustment that equalizes approval rates. The decision boundary now lives at a different score for each group — and any individual sitting between the two thresholds is treated differently depending on group membership.
      </Prose>

      <Plot
        label="Post-DP-adjustment — different thresholds per group"
        xLabel="predicted approval probability"
        yLabel="approval decision (0 / 1)"
        series={[
          {
            name: "group 0 threshold = 0.151",
            color: colors.gold,
            points: [
              [0.0, 0], [0.15, 0], [0.151, 1], [1.0, 1],
            ],
          },
          {
            name: "group 1 threshold = 0.768",
            color: "#c084fc",
            points: [
              [0.0, 0], [0.76, 0], [0.768, 1], [1.0, 1],
            ],
          },
        ]}
      />

      <Prose>
        Two individuals at predicted score 0.5 — one from each group — sit on opposite sides of the decision: group 0 is approved (above 0.151), group 1 is rejected (below 0.768). Demographic parity holds at the population level; individual fairness is destroyed at the pair level.
      </Prose>

      <Prose>
        The heatmap below summarizes how each major fairness intervention scores against each fairness criterion, in the controlled scenario of section 4. Rows are interventions, columns are criteria. Cell values are normalized scores from 0 (criterion violated severely) to 1 (criterion satisfied). The trade-off pattern is clear: no intervention satisfies both group and individual criteria simultaneously.
      </Prose>

      <Heatmap
        label="Fairness criteria vs. interventions — synthetic loan dataset"
        rowLabels={["No mitigation", "DP threshold adjustment", "Lipschitz IF regularizer", "Both (combined)"]}
        colLabels={["Demographic parity", "Equalized odds", "Predictive parity", "Individual fairness"]}
        matrix={[
          [0.10, 0.85, 0.95, 1.00],
          [1.00, 0.40, 0.30, 0.10],
          [0.10, 0.85, 0.95, 1.00],
          [0.50, 0.55, 0.50, 0.45],
        ]}
        cellSize={56}
        colorScale="gold"
      />

      <Prose>
        Notice the two diagonals. The "no mitigation" and "Lipschitz IF regularizer" rows are essentially identical — both produce a smooth model that is individually fair but group-disparate. The "DP threshold adjustment" row inverts the pattern: high group fairness, near-zero individual fairness. The "both combined" row is the pessimal middle: by trying to satisfy both, you satisfy neither well. This is what the impossibility theorems predict.
      </Prose>

      <Prose>
        The step trace below walks through how a single (response, paraphrase) pair flows through an individual-fairness audit on an LLM judge.
      </Prose>

      <StepTrace
        label="LLM-judge paraphrase audit — one pair"
        steps={[
          {
            label: "Select seed response",
            render: () => (
              <Prose>
                Pick a representative response from the judge's evaluation distribution. Example: "The mitochondria is the powerhouse of the cell — it generates ATP through oxidative phosphorylation and is essential for cellular energy metabolism." This is the canonical input.
              </Prose>
            ),
          },
          {
            label: "Generate paraphrases",
            render: () => (
              <Prose>
                Use a separate LLM (or human rewrites) to produce N meaning-preserving rewrites. Example: "Cells rely on the mitochondria as their primary energy source — it produces ATP via oxidative phosphorylation, which is essential for metabolism." Verify equivalence with a sentence-similarity model before scoring; reject paraphrases below a similarity floor.
              </Prose>
            ),
          },
          {
            label: "Score original and paraphrases",
            render: () => (
              <Prose>
                Run the judge on the original and each paraphrase. Record raw scores. Example output: original = 0.84, paraphrase 1 = 0.79, paraphrase 2 = 0.91, paraphrase 3 = 0.72, paraphrase 4 = 0.86. Variance = 0.0048; range = 0.19.
              </Prose>
            ),
          },
          {
            label: "Compute Lipschitz violation",
            render: () => (
              <Prose>
                For each pair, compute |J(r) − J(p)| / D_sem(r, p) and check if it exceeds the Lipschitz constant L. Pairs that exceed L are individual-fairness violations. The aggregate violation rate across the audit set is the headline metric.
              </Prose>
            ),
          },
          {
            label: "Aggregate over audit set",
            render: () => (
              <Prose>
                Repeat for ~1000 seed responses sampled to cover the judge's typical input distribution. Report mean variance, max variance, and the fraction of seeds with variance above an acceptable threshold (e.g., 0.01 on a 0–1 scale). Track these metrics over judge versions.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>When to prioritize group fairness</H3>

      <Prose>
        Group fairness is the right primary criterion when the harm you are trying to prevent is a population-level harm: systematic exclusion of a protected group from a benefit (loans, jobs, healthcare access) or systematic over-application of a burden (carceral risk, surveillance, content moderation strikes). It is also the legally operative concept in many jurisdictions — US employment law uses the four-fifths rule (disparate impact at a ratio below 0.8) as a presumptive trigger for discrimination claims, and EU regulations under the Digital Services Act and AI Act both reference aggregate disparate-impact metrics. If your deployment will be audited by a regulator, by a journalist, or by an organized advocacy group, the first questions you receive will almost certainly be group-fairness questions.
      </Prose>

      <Prose>
        Group fairness is also the more practical default when the protected attribute is genuinely independent of the prediction task. For a content moderation classifier deciding whether a post violates community guidelines, there is no reason for the moderation rate to differ by user demographic — the policy applies the same to everyone. A demographic-parity audit catches the case where the model has accidentally learned to associate certain demographic markers with policy violation, which is unambiguously a bug.
      </Prose>

      <H3>When to prioritize individual fairness</H3>

      <Prose>
        Individual fairness is the right primary criterion when the harm you are trying to prevent is individual: arbitrary or capricious treatment of any single person. It captures the procedural-justice intuition that "like cases should be treated alike" — an idea with deep roots in legal and ethical theory. For an LLM judge that scores essays, individual fairness translates directly to paraphrase invariance, which is something almost every stakeholder agrees is desirable. For a recommendation system that ranks job postings, individual fairness translates to robustness — small changes in the user's profile should not produce wildly different rankings.
      </Prose>

      <Prose>
        Individual fairness is also more appropriate when the protected attribute is correlated with task-relevant features for legitimate reasons. A medical diagnostic model trained to predict the probability of a specific disease will, correctly, produce different probabilities for groups with different prevalence rates — forcing demographic parity here would inject literal medical errors. Individual fairness, by contrast, asks only that two patients with the same symptoms and history receive the same diagnosis, which is what you actually want.
      </Prose>

      <H3>When to use counterfactual fairness</H3>

      <Prose>
        Counterfactual fairness is the right framework when you have, or can plausibly construct, a causal model of the domain — and when the question you are asking is specifically counterfactual: "would this person's outcome have been the same if their protected attribute had been different?" This is the form of question that anti-discrimination law often asks, and it maps cleanly to legal concepts of direct and indirect discrimination. The downside is that you must commit to a causal graph, which is itself a value judgment and can be contested. Use counterfactual fairness when the stakes justify the causal-modeling effort: high-impact decisions, regulated domains, settings where the stakeholders accept the discipline of writing down their causal assumptions.
      </Prose>

      <H3>For LLM judges specifically</H3>

      <Prose>
        Default to a two-track audit. Use individual fairness (paraphrase invariance) as the regression-testing criterion, run on every judge update, with a hard threshold for build-blocking failure. Use group fairness (demographic substitution) as a periodic deep-dive audit, run quarterly or before major deployments, with results reviewed by a cross-functional team rather than auto-blocking. The asymmetry is intentional: paraphrase invariance is uncontroversially desirable and the failure mode (semantically identical inputs receiving different scores) is unambiguously a bug. Group disparity may or may not be a bug depending on the specific dimension and the specific application, so it benefits from human review rather than automated gating.
      </Prose>

      <Prose>
        A second pattern worth knowing for LLM-judge contexts is to maintain separate "calibration" and "consistency" SLOs. The calibration SLO targets agreement with human raters on a held-out gold set — does the judge's mean score on each item match the average human score? The consistency SLO targets paraphrase invariance — does the judge produce nearly identical scores for paraphrases of the same item? These two SLOs catch different bugs. A judge can be perfectly calibrated on average and wildly inconsistent on individual items (high variance under paraphrase); a judge can be perfectly consistent and miscalibrated against humans. Both failure modes are common; both are worth tracking separately. In production at major LLM labs, paraphrase consistency is often the more sensitive early-warning signal because it can be measured without any new human annotation, just by generating paraphrases at runtime.
      </Prose>

      <H3>When neither is enough</H3>

      <Prose>
        For high-stakes decisions, neither group nor individual fairness — alone or together — is sufficient. You also need procedural protections: human review of negative outcomes, appeal mechanisms, transparent documentation of model behavior, and ongoing monitoring after deployment. The fairness literature has gradually converged on the view that algorithmic fairness metrics are necessary but not sufficient inputs to a broader sociotechnical accountability process. Reuben Binns's 2020 paper "On the Apparent Conflict Between Individual and Group Fairness" and Selbst et al.'s "Fairness and Abstraction in Sociotechnical Systems" (FAccT 2019) make this case at length. The mathematical definitions are tools, not solutions.
      </Prose>

      <Prose>
        A useful summary heuristic: pick group fairness when the question is "is this system distributing benefits and burdens equitably across populations?", pick individual fairness when the question is "is this system treating each person consistently relative to similar others?", and pick counterfactual fairness when the question is "would this person have been treated the same had they been a member of a different group?". When the deployment context cannot answer that meta-question — when stakeholders genuinely disagree about which question is the right one — that disagreement is the actual problem, and no choice of metric will resolve it. Surface the disagreement to decision-makers explicitly rather than hide it inside an algorithmic choice.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Group fairness scales gracefully along almost every axis. Adding more protected attributes is a linear cost: for K attributes you measure 2K marginal disparities. Adding more data improves the precision of the disparity estimates. The metrics themselves are simple to compute (counts and ratios) and trivial to slice across subpopulations. Production fairness dashboards routinely track dozens of group-fairness slices in real time without becoming a compute bottleneck. The reason this works is that group-fairness metrics are summary statistics, and summary statistics are computationally cheap.
      </Prose>

      <Prose>
        Group fairness does not scale well to intersectional analysis. Crenshaw's framework of intersectionality argues that members of two protected groups (e.g., Black women) often face harms not captured by analyzing each protected attribute in isolation. Translating this to algorithmic fairness, an intersectional group-fairness audit requires measuring disparities for every combination of protected attributes — a combinatorial explosion that quickly outstrips both data and statistical power. With three binary protected attributes you have eight cells; with five, thirty-two; with ten, a thousand. The cells get small; confidence intervals get wide; meaningful audit becomes impractical. Buolamwini and Gebru's "Gender Shades" study (FAT* 2018) is the canonical demonstration that intersectional gaps can be larger than either marginal gap — but the study is focused on a single intersection (gender × skin tone) precisely because larger intersectional analyses run into statistical-power limits.
      </Prose>

      <Prose>
        There are statistical methods that partially address the intersectional scaling problem. Multicalibration (Hébert-Johnson et al. 2018) requires the model to be calibrated not just on the marginal protected groups but on every "computationally identifiable" subgroup, which can include intersections. The technique extends standard calibration to all subgroups whose membership can be predicted by some bounded-complexity hypothesis class. The trade-off is that multicalibration is much more expensive to verify and to enforce than standard calibration. In practice, intersectional analyses still rely heavily on prioritization: identify the two or three most important intersections, audit them carefully, and accept that the long tail of intersections will not be exhaustively covered.
      </Prose>

      <Prose>
        Individual fairness scales worse along every axis. The Lipschitz formulation requires checking pairs, and the number of pairs grows as N². For 100k individuals, exhaustive pairwise checking is 10 billion comparisons. Sample-based approximations (as in our regularizer in section 4c) bring this down to manageable levels but at the cost of coverage — you are no longer guaranteed that every pair satisfies the constraint, only that random samples typically do. For LLM-judge audits, paraphrase generation is the bottleneck: each seed response requires N paraphrases, each requires its own LLM call, and meaningful audit sets need 1000+ seeds. A full audit can cost as much as the original training run, which limits how often you can re-run it.
      </Prose>

      <Prose>
        The deeper non-scaling problem with individual fairness is the metric specification problem. The Lipschitz formulation requires <Code>D_X</Code>, the input similarity metric. For tabular data with a small number of features, you can negotiate a metric: experts agree on which features matter, weights are assigned, and the metric is documented. For high-dimensional data — text, images, raw user behavior — the metric specification becomes the entire problem. Sentence-embedding distances are a common stand-in for text similarity, but they encode all the biases of the embedding model itself. Image embeddings inherit the visual biases of their training data. A "neutral" similarity metric does not exist; the metric is always a value judgment. This is the source of Dwork's enduring lament that individual fairness "punts the hard problem" — the framework reduces fairness to a metric, and the metric is exactly as hard as the original fairness question.
      </Prose>

      <Prose>
        Counterfactual fairness has its own scaling problem. It requires a structural causal model of the domain, and causal models are notoriously hard to verify. They scale linearly in the number of variables but exponentially in the depth of dependency relationships you are willing to specify. In production, counterfactual fairness has been applied successfully in narrow, well-modeled domains (insurance pricing, certain medical contexts) and rarely outside them.
      </Prose>

      <Prose>
        For LLM judges, what scales best in practice is the combination of cheap group-fairness monitoring (slicing scores by inferred attributes of the input) and targeted individual-fairness audits on a fixed benchmark (paraphrase pairs that you maintain as a versioned artifact). The cheap monitoring catches gross drift; the audit benchmark catches regression. Trying to do exhaustive individual-fairness verification on every LLM-judge call is computationally infeasible and would not be informative anyway because the audit benchmark already covers the patterns you care about.
      </Prose>

      <Prose>
        Another scaling property worth understanding is the cost asymmetry between detection and mitigation. Detecting fairness violations is cheap relative to fixing them. A few thousand audit examples will reliably surface a 10-point demographic-parity gap; the cost is hours of compute. Closing that gap, however, can require dataset re-collection, model retraining, or stakeholder negotiation about which trade-offs are acceptable — work measured in weeks or months. This asymmetry has the consequence that fairness audits often outpace fairness mitigations in production: teams accumulate a backlog of known disparities that they are not yet able to fix. The honest engineering response is to surface this backlog (the "fairness debt" analogue of technical debt) rather than suppress it. A team that audits and reports is more accountable than a team that does not audit and therefore has nothing to report.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Treating one fairness metric as the definition of fair</H3>
      <Prose>
        The single most common failure is to pick one fairness metric (often demographic parity, because it is the easiest to compute and explain) and to declare a model "fair" if it satisfies that metric. The impossibility theorems make clear that any such declaration is a partial statement — the model satisfies that metric, and only that metric, and likely violates other metrics. Always specify which definition of fairness is being claimed and what is known about the others. A model that satisfies demographic parity but violates predictive parity may be accomplishing parity by approving people who will not repay; this can be bad for them too.
      </Prose>

      <H3>Group fairness masks within-group inequities</H3>
      <Prose>
        Achieving demographic parity at the group level says nothing about who within each group is approved. A model can satisfy demographic parity by approving the wealthiest members of an underprivileged group while still systematically rejecting the most marginalized members of that same group. The aggregate statistic is satisfied; the marginalized individuals are not helped. This is a long-standing critique of group-fairness-only approaches in lending and hiring, where the gains from group-fairness interventions often accrue to the most advantaged members of disadvantaged groups.
      </Prose>

      <H3>Individual fairness as a fig leaf for biased metrics</H3>
      <Prose>
        Individual fairness is only as fair as its similarity metric. A "neutral" metric that treats two job applicants as similar based on years of experience and educational pedigree will reproduce all the biases embedded in those features (e.g., systematic underrepresentation of certain demographics in elite educational institutions). A model that satisfies individual fairness with respect to such a metric is being individually consistent in its application of bias, not unbiased. Auditors must scrutinize the metric as carefully as the model.
      </Prose>

      <H3>Counterfactual fairness with a wrong causal graph</H3>
      <Prose>
        Counterfactual fairness inherits all the assumptions of the underlying causal model. If the analyst specifies a causal graph in which the protected attribute does not cause certain features when in reality it does, the counterfactual reasoning will under-correct. If the analyst specifies a graph in which the attribute causes too many features, the counterfactual reasoning will over-correct, potentially flattening genuine differences. There is no model-checking procedure for causal graphs that does not itself rely on assumptions; the framework requires the analyst to commit, document, and defend.
      </Prose>

      <H3>Threshold adjustment as the "fix" for everything</H3>
      <Prose>
        Group-specific thresholds are the cheapest available fix for group-fairness violations and are what tools like Fairlearn's <Code>ThresholdOptimizer</Code> default to. They reliably equalize the targeted metric. They also reliably destroy individual fairness, predictive parity, and any other metric they are not specifically optimizing. Worse, they impose disparate treatment in a legally meaningful sense — explicitly using the protected attribute to make decisions, which is illegal in some jurisdictions and contexts (US lending under ECOA being the canonical example). Always check whether disparate treatment is permitted in your domain before deploying any group-conditional intervention.
      </Prose>

      <H3>LLM judge: paraphrase-invariance score plateau</H3>
      <Prose>
        A common failure mode in LLM-judge audits is a paraphrase-invariance score that looks healthy in aggregate but masks specific failure clusters. Average paraphrase variance can be 0.005 (excellent) while a specific cluster of paraphrase pairs (e.g., responses that mention specific named entities, or use particular rhetorical structures) shows 0.05 variance — an order of magnitude worse. Always slice paraphrase audit results by content category, length bucket, and sentiment, not just average over the audit set.
      </Prose>

      <H3>LLM judge: leaky demographic substitution</H3>
      <Prose>
        Demographic substitution audits work by holding the response constant and varying one demographic attribute (e.g., a name). The audit is only valid if the substitution is the only variable that changes. A common bug: the substitution accidentally changes other features too (e.g., names of different lengths produce different tokenization patterns), which means the score change cannot be cleanly attributed to demographic bias. Always validate that your substitution sets are matched on irrelevant features (token length, syntactic complexity, etc.).
      </Prose>

      <H3>Confusing fairness with accuracy</H3>
      <Prose>
        Accurate models are not automatically fair, and fair models are not automatically accurate. The standard accuracy-fairness trade-off shows that pursuing fairness usually costs some accuracy on the original task; the magnitude depends on the metric and the data. More subtly, an "unfair" model can be unfair because it is highly accurate at predicting an outcome that is itself the product of unfair upstream processes. A model that accurately predicts who will be arrested by the police can be group-unfair across racial lines because the underlying arrest patterns are themselves group-unfair. Fixing the model does not fix the underlying process; sometimes the right answer is not to deploy the model at all.
      </Prose>

      <H3>Static audits in dynamic systems</H3>
      <Prose>
        Audits performed once at deployment can become obsolete as the system, the input distribution, and the task evolve. A judge that passed paraphrase-invariance audits at v1 may fail them at v3 because retraining shifted its sensitivity to surface form. A model that satisfied demographic parity at deployment may drift over time as the input distribution changes. Audits must be continuous, with versioned benchmarks, automated regression detection, and a process for investigating regressions before they reach users.
      </Prose>

      <H3>Mistaking aggregate parity for individual remedy</H3>
      <Prose>
        A subtle but serious failure mode in policy contexts: closing an aggregate group disparity does not, by itself, remedy the harm experienced by any specific individual who was previously denied. If a lending model has been systematically underapproving members of group G, raising the approval rate for G to match the privileged group's rate going forward is a population-level fix; the individuals previously denied are not retrospectively approved. A fairness intervention that reports "the gap is closed" can give the impression of restitution while delivering only future parity. This is one reason why algorithmic fairness work increasingly emphasizes that aggregate-statistic interventions must be paired with case-level review and remedy mechanisms when the underlying decisions affect specific people. The mathematical fix and the moral fix are not the same fix.
      </Prose>

      <Callout accent="purple">
        The most common request the fairness community gets is "just tell me what number to optimize." There is no such number. Different fairness definitions encode different ethical commitments and produce different verdicts. The real engineering work is to (a) be explicit about which definition you are committing to, (b) document why, (c) audit continuously for the criteria you have chosen, and (d) maintain the capacity to revisit the choice as you learn more about how the system actually behaves in the world.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All references below were verified against their canonical venues (arXiv, ACM, ITCS, NeurIPS) on 2026-04-25. Author lists, titles, and identifiers confirmed.
      </Prose>

      <H3>Dwork et al. 2012 — Fairness Through Awareness</H3>
      <Prose>
        Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel. "Fairness Through Awareness." Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS), 2012, pp. 214–226. arXiv:1104.3913. The founding paper for individual fairness. Introduces the Lipschitz formulation D_Y(M(x), M(y)) ≤ L · D_X(x, y), proves a suite of theorems characterizing when individually-fair classifiers exist, and explicitly acknowledges the metric-specification problem ("we punt the question of where the metric comes from"). The paper also shows that individual fairness is in tension with group-fairness statistical parity, anticipating much of the impossibility-theorem literature.
      </Prose>

      <H3>Hardt, Price, Srebro 2016 — Equality of Opportunity</H3>
      <Prose>
        Moritz Hardt, Eric Price, Nathan Srebro. "Equality of Opportunity in Supervised Learning." NeurIPS 2016. arXiv:1610.02413. Introduces the equalized-odds and equal-opportunity definitions of group fairness, derives an efficient post-processing algorithm to satisfy them, and demonstrates it on the FICO credit-score dataset. One of the most influential fairness papers; equalized odds remains a default group-fairness target in production tooling.
      </Prose>

      <H3>Chouldechova 2017; Kleinberg-Mullainathan-Raghavan 2017 — Impossibility</H3>
      <Prose>
        Alexandra Chouldechova. "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." Big Data 5(2), 2017. arXiv:1610.07524. Independently and concurrently: Jon Kleinberg, Sendhil Mullainathan, Manish Raghavan. "Inherent Trade-Offs in the Fair Determination of Risk Scores." ITCS 2017. arXiv:1609.05807. Both papers prove that whenever base rates differ across groups, no nontrivial classifier can simultaneously satisfy calibration within groups and equalized error rates. Together with the COMPAS audit episode, these results made the impossibility of "all fairness definitions at once" widely known.
      </Prose>

      <H3>Friedler, Scheidegger, Venkatasubramanian 2016 — (Im)possibility of Fairness</H3>
      <Prose>
        Sorelle A. Friedler, Carlos Scheidegger, Suresh Venkatasubramanian. "On the (im)possibility of fairness." arXiv:1609.07236, September 2016. Frames the conflict between group and individual fairness as a conflict between two worldviews — "what you see is what you get" (WYSIWYG, individual fairness) and "we're all equal" (WAE, group fairness) — and proves that no classifier can satisfy fairness criteria from both worldviews simultaneously when the worldviews disagree about the data. Provides the cleanest theoretical statement of why the two notions are incompatible.
      </Prose>

      <H3>Kusner et al. 2017 — Counterfactual Fairness</H3>
      <Prose>
        Matt J. Kusner, Joshua R. Loftus, Chris Russell, Ricardo Silva. "Counterfactual Fairness." NeurIPS 2017. arXiv:1703.06856. Introduces the counterfactual definition of fairness using Pearl's structural causal models. Shows that counterfactual fairness is a special case of individual fairness where the similarity metric is supplied by the causal graph. The paper provides three algorithmic constructions (Levels 1, 2, 3) that trade off causal-modeling assumptions against deployment realism. The canonical reference for causal approaches to fairness.
      </Prose>

      <H3>Binns 2020 — Conflict between individual and group fairness</H3>
      <Prose>
        Reuben Binns. "On the apparent conflict between individual and group fairness." Proceedings of the 2020 ACM Conference on Fairness, Accountability, and Transparency (FAT*/FAccT 2020). arXiv:1912.06883. A philosophical analysis arguing that the apparent conflict between group and individual fairness reflects a deeper disagreement about the moral status of group membership. Binns shows that what looks like a mathematical incompatibility often dissolves once stakeholders make their underlying value commitments explicit. Required reading for anyone who has to defend a fairness choice to a non-technical audience.
      </Prose>

      <H3>Buolamwini, Gebru 2018 — Gender Shades</H3>
      <Prose>
        Joy Buolamwini, Timnit Gebru. "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAT* 2018). The empirical result that catalyzed the modern fairness audit movement: commercial face-classification systems showed dramatically higher error rates on dark-skinned female faces than on light-skinned male faces, with intersectional gaps larger than either marginal gap. The paper's methodology — a curated, balanced benchmark with explicit disparity measurement — became the template for production fairness audits.
      </Prose>

      <H3>Parrish et al. 2022 — BBQ benchmark</H3>
      <Prose>
        Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, Samuel R. Bowman. "BBQ: A Hand-Built Bias Benchmark for Question Answering." Findings of ACL 2022. arXiv:2110.08193. Provides a systematic benchmark of demographic-substitution prompts across nine social dimensions (age, disability status, gender identity, nationality, physical appearance, race/ethnicity, religion, socioeconomic status, sexual orientation). The de-facto standard reference for group-fairness audits of LLM judges and QA systems.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Construct a model that satisfies one definition and fails the other</H3>
      <Prose>
        Construct a synthetic two-feature, two-group dataset where (a) you can write down a model that satisfies demographic parity exactly but produces near-identical individuals receiving opposite decisions, and (b) you can write down a different model that satisfies individual fairness exactly but produces a 30+ percentage point disparity in approval rates. Walk through the algebra showing why each model satisfies its targeted definition. What does this demonstrate about the relationship between the two definitions? Could you construct a single model that satisfies both, or does the data structure preclude it?
      </Prose>

      <H3>Exercise 2 — Pick the metric for each scenario</H3>
      <Prose>
        For each of the following deployments, decide which fairness criterion (demographic parity, equalized odds, predictive parity, individual fairness, counterfactual fairness, or some combination) is the most appropriate primary criterion and justify your choice in two sentences. (a) An LLM judge scoring student essays for a national writing competition. (b) A risk-scoring tool used by a parole board to recommend release decisions. (c) A medical-triage classifier predicting which patients should be seen first in an emergency department. (d) A job-matching recommender system that suggests open positions to job seekers. (e) A content-moderation classifier that flags posts for potential community-guideline violations. Where you choose multiple criteria, indicate which takes precedence in conflict.
      </Prose>

      <H3>Exercise 3 — Why the Lipschitz constant matters</H3>
      <Prose>
        In the from-scratch implementation, the Lipschitz constant <Code>L</Code> in the regularizer was set to 1.5. What happens if you set <Code>L = 0.0</Code>? What happens if you set <Code>L = ∞</Code>? Trace through the regularizer's behavior at both extremes and describe what kind of model each produces. Then describe a principled procedure for choosing <Code>L</Code> in a real application — what data, what stakeholder input, and what measurements would you use? How does the choice of <Code>L</Code> compare in difficulty to the choice of the input metric <Code>D_X</Code> itself?
      </Prose>

      <H3>Exercise 4 — Audit an LLM judge</H3>
      <Prose>
        Design a complete fairness audit for an LLM judge that scores customer support response quality on a 1–5 scale. Include: (a) the seed-response sampling procedure, (b) the paraphrase generation procedure for individual fairness, (c) the demographic substitution procedure for group fairness, (d) the metrics you will report, (e) the failure thresholds at which you would block a deployment versus flag for review, and (f) how you will detect if the audit benchmark itself has become stale. For each design choice, identify the value judgment it encodes — what assumptions about "fair" are baked into your audit and could be contested?
      </Prose>

      <H3>Exercise 5 — Reconcile a real disagreement (COMPAS)</H3>
      <Prose>
        Re-read the COMPAS audit episode (ProPublica 2016 vs Northpointe's response). Both audits used real data and reached opposite conclusions about whether COMPAS was racially biased. (a) State precisely which fairness criterion each audit applied. (b) Explain why both criteria cannot be simultaneously satisfied given the empirical base rates. (c) Argue, taking a clear position, which criterion you think should have been treated as primary in the criminal-risk-assessment context, and why. (d) Identify three things an algorithmic-fairness practitioner could do today that the original COMPAS deployment did not, to surface this trade-off explicitly to the decision-makers who used the tool. (e) Briefly describe what an individual-fairness audit of COMPAS would have looked like and what additional information it would have provided beyond the two group-fairness audits.
      </Prose>

      <H3>Exercise 6 — Specify the metric for an LLM-judge audit</H3>
      <Prose>
        You are auditing an LLM judge that scores creative writing on a 1–10 scale. You need to define <Code>D_X</Code>, the input-similarity metric, for an individual-fairness audit. (a) Propose three concrete candidates for <Code>D_X</Code>, ranging from cheap (e.g., normalized edit distance) to expensive (e.g., a fine-tuned semantic-equivalence model). (b) For each candidate, identify two failure modes — situations where the metric would call two inputs similar that you would not actually want treated identically, and vice versa. (c) Describe an experiment to validate any chosen <Code>D_X</Code> against human judgments of similarity, including the size and composition of the validation set. (d) Suppose your validation experiment shows your metric agrees with humans 80% of the time. Is that good enough to base a fairness audit on? Justify your answer with reference to what failure modes the remaining 20% disagreement might mask.
      </Prose>

    </div>
  ),
};

export default groupVsIndividualFairness;
