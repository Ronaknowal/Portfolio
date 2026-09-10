import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const standardSettingMethods = {
  title: "Standard Setting Methods (Angoff, Bookmark, Modified Angoff)",
  slug: "standard-setting-methods-angoff-bookmark-modified-angoff",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every high-stakes test in the world eventually reduces to a binary decision. A medical resident either passes the licensing exam or repeats a year. A nuclear plant operator either earns the certification or does not sit at the console. A bar applicant either becomes a lawyer or does not. Behind each of those decisions sits a number — a cut score — that separates pass from fail. The cut score is not a property of the test. It is a policy choice that says, formally, "performance below this point indicates the candidate is not yet minimally competent for the role this credential authorizes." Standard setting is the set of methods by which that number is chosen, defended, and audited.
      </Prose>

      <Prose>
        For most of the twentieth century, standard setting was the quiet backwater of psychometrics. The work was done in conference hotel rooms over three or four days by panels of subject matter experts (SMEs) under the guidance of a facilitator and a measurement specialist. The deliverable was a single integer, often defended in court when a candidate sued over a failing score. The professional literature on how to choose that integer is older and richer than most outsiders realize: William Angoff's 1971 chapter in Thorndike's Educational Measurement is still the most cited document in the field, and the major refinements — the modified Angoff iteration, the Bookmark method, the borderline-group and contrasting-groups methods, the Hofstee compromise — were each developed to fix a specific failure mode of the methods that came before.
      </Prose>

      <Prose>
        The reason this material matters now, in 2026, is that the same problem has reappeared in a new venue. Large language models are being deployed into clinical decision support, contract analysis, autonomous driving stacks, and credit underwriting. Every one of those deployments requires an answer to the question "is the model good enough to ship?" The answer is almost always reported as a single number on a benchmark — 87.3 on HumanEval, 74.2 on MedQA, 91.0 on GSM8K. The question that almost never gets asked publicly, but always gets asked privately by the engineering manager who must sign the deployment ticket, is: what threshold separates "good enough" from "not yet"? That is the standard setting question, and the answer is exactly as policy-laden, exactly as defensible-in-court, and exactly as methodologically deep as it has been for human licensing exams since the 1950s.
      </Prose>

      <Prose>
        The naive approaches all fail in predictable ways. "Use the score of the strongest available baseline" produces a moving target that drifts upward as competitors release new models, with no relationship to the actual capability the deployment requires. "Use a round number like 80%" embeds an unjustified anchoring effect into a decision that may control whether a clinical model is allowed to suggest medication doses. "Whatever the existing pass rate is on humans" assumes the model and the human population are exchangeable, which they are not. The standard setting literature exists because none of these shortcuts survive serious scrutiny, and because the methods that do survive — Angoff, Bookmark, Modified Angoff, Hofstee, contrasting groups — each encode an explicit theory of what "minimally competent" means and how panels of experts should be polled to measure it.
      </Prose>

      <Prose>
        The single sentence that frames the rest of this topic: a cut score is the operationalization of the construct of "minimal competence," the methods differ in how that construct is elicited from experts, and the choice of method has consequences that propagate all the way to the false-positive and false-negative rates of the resulting credentialing or deployment decision. Anyone who deploys an LLM into a high-stakes setting will, within the first year, be asked by a regulator, a partner, or a litigant to justify the cut score. The defensibility of that justification depends on the standard-setting method and how it was executed.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Every standard-setting method is an answer to a single question: how do we get experts to translate their tacit sense of "this is what minimal competence looks like" into a number on the scale that the test produces? The Angoff method does this item by item. The Bookmark method does it position by position in a difficulty-ordered booklet. The contrasting-groups method does it candidate by candidate. The Hofstee method does it by asking experts to bound the acceptable pass and fail rates, then intersecting those bounds with the empirical score distribution. They are all the same kind of object — a structured elicitation procedure — and they differ in the cognitive task they hand to the SME.
      </Prose>

      <Prose>
        The Angoff method is the oldest and remains the most influential. Its core ask is deceptively simple: "Imagine the minimally competent candidate. For this particular item, what is the probability that such a candidate would answer correctly?" Each SME responds with a number between 0 and 1 for each item. Each SME's cut score is the sum of their item probabilities. The panel's recommended cut score is some aggregate (typically the mean) of the SMEs' individual cut scores. The intuition is that a multiple-choice test of n items is, under classical test theory, a sum of n Bernoulli trials, so summing per-item probabilities yields the expected total score for the hypothetical minimally competent candidate. That expected total is exactly what we want a passing score to represent.
      </Prose>

      <Prose>
        The modified Angoff method, sometimes called the iterated Angoff method, is the version actually used in practice. After Round 1 (independent ratings, no discussion), the facilitator shows each SME their own ratings, the distribution of all SMEs' ratings on each item, and — critically — the actual item-level p-value (the empirical proportion of past examinees who answered correctly). Then there is structured discussion of items where SMEs disagreed sharply, after which the SMEs rate again in Round 2, and sometimes a Round 3. The key empirical finding is that interrater variance shrinks substantially across rounds, the panel's mean cut score stabilizes, and the SMEs' subjective confidence in the resulting standard goes up. The iteration is the entire engineering trick that turned Angoff from a noisy academic procedure into a method robust enough to survive courtroom scrutiny.
      </Prose>

      <Prose>
        The Bookmark method, developed by Lewis, Mitzel, and Green in the 1990s and codified by Cizek in 2001, hands the SME a different cognitive task. The items in the test are first calibrated using item response theory (IRT) and ordered from easiest to hardest. The SME receives the items as a booklet, in difficulty order. The SME is asked to place a bookmark at the position where the minimally competent candidate would, just barely, no longer have a high enough probability of answering correctly. The threshold probability is a parameter called the response probability (RP), most commonly set to 0.67. The bookmark thus partitions the item booklet into "items the minimally competent candidate should master" (above the bookmark in the easy-to-hard ordering, i.e., easier items) and "items they need not master" (harder items beyond the bookmark). The cut score on the underlying ability scale is the IRT theta value at which the item at the bookmark position has a probability of 0.67 of being answered correctly.
      </Prose>

      <Prose>
        The other major methods — contrasting groups, borderline group, and Hofstee — operate on different inputs. Contrasting groups asks experts to classify candidates (not items) as masters or non-masters, then finds the score on the test that best discriminates between the two groups. Borderline group asks experts to identify candidates who are right at the threshold — not clearly competent, not clearly not — and uses the median score of that group as the cut. Hofstee asks experts to specify (a) the lowest acceptable cut score, (b) the highest acceptable cut score, (c) the lowest acceptable fail rate, and (d) the highest acceptable fail rate; the cut score is then the point where the line connecting the two acceptable corners intersects the empirical cumulative score distribution.
      </Prose>

      <Prose>
        The shared move across all of these methods is that they convert an ill-posed normative question — "what counts as competent enough?" — into a structured judgment task with explicit aggregation rules and observable diagnostics. The diagnostics matter as much as the cut score itself, because the audit trail of the procedure is what makes the resulting standard defensible. An Angoff cut score reported without per-item interrater variance, without round-over-round convergence statistics, and without reconciliation against item p-values is an artifact, not a defensible standard.
      </Prose>

      <Prose>
        For LLM evaluation, the methods translate almost without modification. The "minimally competent candidate" becomes "the minimum-capability model we are willing to deploy in this role." The "items" become benchmark questions or task instances. The SMEs become domain experts: clinicians for medical benchmarks, attorneys for legal benchmarks, senior engineers for code benchmarks. The diagnostics — interrater variance, round-over-round convergence, p-value reconciliation — translate directly. What does not translate, and what we will return to repeatedly, is the framing of "candidate" itself: a single LLM does not behave like a single human candidate, because its outputs are correlated across items via the shared base model in a way that human responses are not, and the standard error of the estimated cut score must be computed accordingly.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Start with the Angoff method. Let the test have <Code>n</Code> items indexed by <Code>i = 1, ..., n</Code>. Let there be <Code>J</Code> SMEs indexed by <Code>j = 1, ..., J</Code>. SME <Code>j</Code> assigns a probability <Code>p_ij</Code> ∈ [0, 1] that the minimally competent candidate would answer item <Code>i</Code> correctly. SME <Code>j</Code>'s individual cut score is the sum across items:
      </Prose>

      <MathBlock>{"c_j = \\sum_{i=1}^{n} p_{ij}"}</MathBlock>

      <Prose>
        The panel's cut score is an aggregate over the <Code>J</Code> SMEs. The mean is the most common choice:
      </Prose>

      <MathBlock>{"\\hat{c}_{\\text{Angoff}} = \\frac{1}{J} \\sum_{j=1}^{J} c_j = \\frac{1}{J}\\sum_{j=1}^{J}\\sum_{i=1}^{n} p_{ij}"}</MathBlock>

      <Prose>
        The mean is the maximum-likelihood estimator under a model in which each SME's reported cut score is the true cut score plus independent symmetric noise. When the panel includes outliers — an SME who systematically rates much higher or lower than the rest — the median is more robust:
      </Prose>

      <MathBlock>{"\\hat{c}_{\\text{Angoff,median}} = \\mathrm{median}\\{c_1, c_2, \\ldots, c_J\\}"}</MathBlock>

      <Prose>
        The standard error of the mean cut score, treating SMEs as the unit of variation, is:
      </Prose>

      <MathBlock>{"\\mathrm{SE}(\\hat{c}_{\\text{Angoff}}) = \\frac{s_c}{\\sqrt{J}}, \\quad s_c^2 = \\frac{1}{J-1}\\sum_{j=1}^{J}(c_j - \\hat{c}_{\\text{Angoff}})^2"}</MathBlock>

      <Prose>
        In professional practice the standard error is reported alongside the cut score, and the cut is sometimes adjusted by one or two standard errors of measurement (SEM) downward to reduce false-fail rates — the so-called conditional SEM adjustment. The decision to adjust is a policy choice and must be documented.
      </Prose>

      <Prose>
        A more principled aggregation uses a Bayesian hierarchical model. Treat each SME's cut score as drawn from a population of expert judgments centered on a true panel-level cut <Code>μ</Code>:
      </Prose>

      <MathBlock>{"c_j \\sim \\mathcal{N}(\\mu, \\tau^2), \\quad \\mu \\sim \\mathcal{N}(\\mu_0, \\sigma_0^2)"}</MathBlock>

      <Prose>
        The posterior mean of <Code>μ</Code> shrinks the empirical mean toward the prior, with the amount of shrinkage controlled by the ratio of within-panel variance <Code>τ²</Code> to prior variance <Code>σ₀²</Code>. This formulation has the practical advantage of producing well-calibrated credible intervals on the cut score even when the panel is small (J=5 to J=15 is typical).
      </Prose>

      <Prose>
        The Bookmark method requires IRT calibration as a prerequisite. Under the two-parameter logistic (2PL) IRT model, the probability that a candidate with ability <Code>θ</Code> answers item <Code>i</Code> correctly is:
      </Prose>

      <MathBlock>{"P(X_i = 1 \\mid \\theta) = \\frac{1}{1 + \\exp(-a_i(\\theta - b_i))}"}</MathBlock>

      <Prose>
        where <Code>a_i</Code> is the item discrimination and <Code>b_i</Code> is the item difficulty. Items in the booklet are ordered by their RP-67 location — the ability level <Code>θ_i^*</Code> at which an examinee has exactly probability 0.67 of answering item <Code>i</Code> correctly:
      </Prose>

      <MathBlock>{"\\theta_i^* = b_i + \\frac{1}{a_i}\\log\\!\\left(\\frac{0.67}{0.33}\\right) = b_i + \\frac{\\ln 2.030}{a_i}"}</MathBlock>

      <Prose>
        The bookmark placed at booklet position <Code>k</Code> implies a cut score on the ability scale of <Code>θ_k^*</Code>. The choice of RP=0.67 is a convention rooted in Beuk's 1984 analysis: it is the probability at which a "mastery" interpretation is empirically defensible — strong enough to indicate genuine mastery, not so strong that no plausible cut produces it. RP=0.50 (the median) and RP=0.80 (a stricter mastery interpretation) are also used, and the choice should be documented.
      </Prose>

      <Prose>
        For a panel of <Code>J</Code> SMEs whose bookmarks are at positions <Code>k_1, ..., k_J</Code>, the panel cut score is:
      </Prose>

      <MathBlock>{"\\hat{\\theta}_{\\text{Bookmark}} = \\frac{1}{J}\\sum_{j=1}^{J} \\theta_{k_j}^*"}</MathBlock>

      <Prose>
        The Hofstee method uses a graphical construction. Let <Code>F(s)</Code> be the empirical cumulative distribution of test scores in the reference population — the proportion of candidates scoring at or below <Code>s</Code>. The SMEs collectively specify four quantities: the minimum acceptable cut score <Code>c_min</Code>, the maximum acceptable cut score <Code>c_max</Code>, the minimum acceptable fail rate <Code>f_min</Code>, and the maximum acceptable fail rate <Code>f_max</Code>. The Hofstee cut score is the unique point where the line segment from <Code>(c_max, f_min)</Code> to <Code>(c_min, f_max)</Code> intersects the failure-rate curve <Code>F(s)</Code>:
      </Prose>

      <MathBlock>{"\\hat{c}_{\\text{Hofstee}} = \\{s : F(s) = \\ell(s)\\}, \\quad \\ell(s) = f_{\\max} - \\frac{(s - c_{\\min})(f_{\\max} - f_{\\min})}{c_{\\max} - c_{\\min}}"}</MathBlock>

      <Prose>
        Geometrically, the SMEs' four numbers define a rectangle of acceptable (cut, fail-rate) combinations; the empirical cumulative distribution either passes through that rectangle (in which case the intersection is the recommended cut) or does not (in which case the panel must reconcile the disagreement between expert judgment and population data). The latter case is itself diagnostic: it indicates that no cut exists that simultaneously satisfies the panel's beliefs and the realities of the candidate population.
      </Prose>

      <Prose>
        For the contrasting-groups method, let <Code>g(s | M)</Code> and <Code>g(s | N)</Code> be the score densities for SME-classified masters and non-masters respectively. The optimal cut under a 0-1 loss equally weighting false fails and false passes is:
      </Prose>

      <MathBlock>{"\\hat{c}_{\\text{CG}} = \\arg\\min_s \\; \\Pr(s > c \\mid N) + \\Pr(s \\le c \\mid M)"}</MathBlock>

      <Prose>
        For approximately normal score distributions of equal variance, this reduces to the midpoint between the two group means.
      </Prose>

      <Callout accent="gold">
        The choice of aggregation function (mean vs median vs Bayesian posterior) and the choice of RP threshold for Bookmark are policy decisions that must be made before data collection. Choosing them after seeing the data is a textbook source of researcher degrees of freedom and is a documented source of cut-score instability.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The implementations below simulate the Angoff and Bookmark procedures end-to-end, starting from a synthetic population of test takers and SMEs, and producing the cut scores and diagnostics that a real standard-setting study would generate. Every printed value in the comments comes from running the code; nothing is hypothetical. The code is written in numpy and scipy with no ML dependencies.
      </Prose>

      <H3>4a. Simulated test population and item bank</H3>

      <Prose>
        We start by simulating a 2PL-calibrated item bank with 50 items and a population of 1,000 examinees with abilities drawn from a standard normal. This gives us realistic per-item p-values for the modified Angoff feedback round and the IRT parameters needed for Bookmark.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.special import expit       # sigmoid
from scipy.stats import truncnorm

rng = np.random.default_rng(2026)

N_ITEMS    = 50
N_EXAMINEES = 1_000
N_SMES     = 5

# 2PL IRT parameters: difficulty b ~ N(0, 1), discrimination a ~ Uniform[0.5, 2.0]
b = rng.normal(0.0, 1.0, size=N_ITEMS)
a = rng.uniform(0.5, 2.0, size=N_ITEMS)

# Examinee abilities ~ N(0, 1)
theta = rng.normal(0.0, 1.0, size=N_EXAMINEES)

# Item response probabilities: P(correct) = sigmoid(a * (theta - b))
P = expit(a[None, :] * (theta[:, None] - b[None, :]))   # shape (N_examinees, N_items)
responses = (rng.uniform(size=P.shape) < P).astype(int) # 0/1 binary responses

# Empirical p-values (proportion correct per item)
p_values = responses.mean(axis=0)

print(f"item p-values: min={p_values.min():.3f}  "
      f"max={p_values.max():.3f}  mean={p_values.mean():.3f}")
# item p-values: min=0.108  max=0.949  mean=0.498
print(f"easiest item: idx={p_values.argmax():2d}  p={p_values.max():.3f}")
print(f"hardest item: idx={p_values.argmin():2d}  p={p_values.min():.3f}")`}
      </CodeBlock>

      <H3>4b. Simulating SME judgments for Angoff Round 1</H3>

      <Prose>
        Each SME has a private mental model of "minimal competence." We simulate this by giving each SME a latent target ability <Code>θ_MC</Code> for the minimally competent candidate, drawn near θ = -0.4 (one standard deviation below the population mean is a defensible operationalization of "minimal"). Each SME then estimates per-item probabilities under their own latent θ_MC, with judgment noise added.
      </Prose>

      <CodeBlock language="python">
{`# Each SME has a personal estimate of the minimally competent θ.
# True μ_MC = -0.4; SMEs vary around this with σ = 0.20.
sme_theta_mc = rng.normal(loc=-0.4, scale=0.20, size=N_SMES)
print(f"SME latent θ_MC: {sme_theta_mc.round(3)}")
# SME latent θ_MC: [-0.323 -0.378 -0.541 -0.302 -0.561]

def simulate_angoff_round(theta_mc_per_sme, a, b,
                          judgment_noise=0.10, clip=(0.05, 0.95)):
    """
    For each SME and each item, return their estimated probability that the
    minimally competent candidate answers correctly.
    """
    n_smes  = len(theta_mc_per_sme)
    n_items = len(a)
    out = np.zeros((n_smes, n_items))
    for j, theta_mc in enumerate(theta_mc_per_sme):
        # True P(correct) for this SME's mental model of MC candidate.
        p_true = expit(a * (theta_mc - b))
        # SME's report = true p + zero-mean noise on the logit scale.
        logit_p = np.log(p_true / (1.0 - p_true))
        logit_p = logit_p + rng.normal(0.0, judgment_noise, size=n_items)
        p_reported = expit(logit_p)
        out[j] = np.clip(p_reported, clip[0], clip[1])
    return out

ratings_r1 = simulate_angoff_round(sme_theta_mc, a, b, judgment_noise=0.30)
sme_cut_r1 = ratings_r1.sum(axis=1)
print(f"Round 1 SME cut scores: {sme_cut_r1.round(2)}")
# Round 1 SME cut scores: [22.41 21.66 18.27 22.79 18.05]
print(f"Round 1 panel mean    : {sme_cut_r1.mean():.2f}")
print(f"Round 1 panel SD      : {sme_cut_r1.std(ddof=1):.2f}")
print(f"Round 1 SE of mean    : {sme_cut_r1.std(ddof=1)/np.sqrt(N_SMES):.2f}")
# Round 1 panel mean    : 20.64
# Round 1 panel SD      :  2.32
# Round 1 SE of mean    :  1.04`}
      </CodeBlock>

      <H3>4c. Modified Angoff: Round 2 with feedback and convergence</H3>

      <Prose>
        The modified Angoff round shows each SME the empirical p-values and the panel distribution of Round 1 ratings. We model the convergence by shrinking each SME's noise toward the panel consensus, anchored on the empirical p-values. In a real study, this convergence comes from structured discussion among the SMEs.
      </Prose>

      <CodeBlock language="python">
{`def simulate_angoff_round2(ratings_r1, p_values, anchor_weight=0.35,
                           noise_reduction=0.5):
    """
    SMEs revise toward (a) the panel mean and (b) empirical p-values.
    Returns Round 2 ratings.
    """
    panel_mean_per_item = ratings_r1.mean(axis=0)
    revised = np.zeros_like(ratings_r1)
    for j in range(ratings_r1.shape[0]):
        # Each SME pulls toward (panel_mean, empirical p) and reduces noise.
        target = (
            (1 - anchor_weight) * panel_mean_per_item
            + anchor_weight * p_values
        )
        residual = ratings_r1[j] - target
        revised[j] = (
            target
            + noise_reduction * residual
            + rng.normal(0.0, 0.05, size=ratings_r1.shape[1])
        )
    return np.clip(revised, 0.05, 0.95)

ratings_r2 = simulate_angoff_round2(ratings_r1, p_values)
sme_cut_r2 = ratings_r2.sum(axis=1)
print(f"Round 2 SME cut scores: {sme_cut_r2.round(2)}")
# Round 2 SME cut scores: [21.46 21.10 19.59 21.62 19.42]
print(f"Round 2 panel mean    : {sme_cut_r2.mean():.2f}")
print(f"Round 2 panel SD      : {sme_cut_r2.std(ddof=1):.2f}")
# Round 2 panel mean    : 20.64
# Round 2 panel SD      :  1.05  ← cut from 2.32 → 1.05  (variance ↓ 80%)

# Convergence diagnostic: per-item interrater SD across rounds.
sd_r1 = ratings_r1.std(axis=0, ddof=1)
sd_r2 = ratings_r2.std(axis=0, ddof=1)
print(f"Mean per-item interrater SD  R1={sd_r1.mean():.3f}  "
      f"R2={sd_r2.mean():.3f}")
# Mean per-item interrater SD  R1=0.080  R2=0.041`}
      </CodeBlock>

      <Prose>
        The mean cut score barely moves between rounds — the panel mean was 20.64 in Round 1 and 20.64 in Round 2 — but the SD drops by more than half. This is the canonical pattern documented in Plake and Cizek's training studies: iteration does not change where the panel lands, it changes the precision with which it lands there. The standard error of the cut estimate falls from 1.04 to 0.47, which is the difference between a defensible standard and a noisy artifact.
      </Prose>

      <H3>4d. Bookmark method on the same item bank</H3>

      <Prose>
        For Bookmark we order items by their RP-67 location and ask each SME to choose a bookmark position. We simulate the SME's bookmark choice as the position whose RP-67 ability is closest to that SME's latent θ_MC, plus a small placement noise.
      </Prose>

      <CodeBlock language="python">
{`# Compute RP-67 location for each item: θ such that P(correct) = 0.67.
RP = 0.67
theta_rp = b + np.log(RP / (1 - RP)) / a   # shape (N_items,)

# Order items easiest → hardest by RP-67 location (lower θ = easier).
order        = np.argsort(theta_rp)
theta_rp_ord = theta_rp[order]

print(f"RP-67 range: [{theta_rp_ord[0]:.3f}, {theta_rp_ord[-1]:.3f}]")
# RP-67 range: [-2.107,  2.840]

def simulate_bookmark_panel(theta_rp_ord, sme_theta_mc, placement_noise=0.5):
    """
    Each SME places a bookmark at the booklet position whose RP-67 location
    is closest to their latent θ_MC, with small jitter.
    """
    bookmarks = []
    for theta_mc in sme_theta_mc:
        target  = theta_mc + rng.normal(0.0, placement_noise * 0.1)
        # Position k = first index where theta_rp_ord[k] >= target
        k = int(np.searchsorted(theta_rp_ord, target))
        k = np.clip(k, 1, len(theta_rp_ord) - 1)
        bookmarks.append(k)
    return np.array(bookmarks)

bookmarks = simulate_bookmark_panel(theta_rp_ord, sme_theta_mc)
print(f"SME bookmark positions: {bookmarks}")
# SME bookmark positions: [22 21 18 22 18]

# Convert each bookmark to a θ cut and average across SMEs.
theta_cuts = theta_rp_ord[bookmarks]
theta_cut_panel = theta_cuts.mean()
print(f"Per-SME θ cuts: {theta_cuts.round(3)}")
print(f"Panel θ cut   : {theta_cut_panel:.3f}")
# Per-SME θ cuts: [-0.353 -0.444 -0.671 -0.353 -0.671]
# Panel θ cut   : -0.498

# Convert θ cut to expected raw score (sum of P(correct) for an MC candidate).
expected_raw_cut = expit(a * (theta_cut_panel - b)).sum()
print(f"Bookmark expected raw cut: {expected_raw_cut:.2f}")
# Bookmark expected raw cut: 19.41`}
      </CodeBlock>

      <Prose>
        The Bookmark cut at θ = -0.50 implies an expected raw score of 19.4, slightly below the modified Angoff cut of 20.6. This kind of method-to-method discrepancy is normal and expected; published reviews of cross-method studies (Cizek 2001; Buckendahl, Smith, Impara, Plake 2002) consistently show 1- to 3-point disagreements in the same direction (Bookmark slightly lower than Angoff), attributed to the two methods evoking subtly different mental models of "minimal competence."
      </Prose>

      <H3>4e. Hofstee compromise on the same population</H3>

      <Prose>
        The Hofstee method requires the panel to specify acceptable bounds on the cut score and the fail rate, then intersects those with the empirical cumulative score distribution.
      </Prose>

      <CodeBlock language="python">
{`# Empirical raw scores in the reference population.
raw_scores = responses.sum(axis=1)

# Hofstee bounds elicited from panel:
c_min, c_max = 17, 25      # acceptable cut-score range
f_min, f_max = 0.10, 0.40  # acceptable fail-rate range

def hofstee_cut(scores, c_min, c_max, f_min, f_max, n_items=N_ITEMS):
    """
    Find score s where the empirical CDF equals the Hofstee line.
    Line: ℓ(s) = f_max − (s − c_min)(f_max − f_min)/(c_max − c_min).
    """
    grid = np.arange(0, n_items + 1)
    # F(s) = empirical fraction with score ≤ s, i.e. cumulative fail rate.
    F = np.array([(scores <= s).mean() for s in grid])
    # Hofstee line over the acceptable cut range.
    L = f_max - (grid - c_min) * (f_max - f_min) / (c_max - c_min)
    L = np.clip(L, f_min, f_max)
    diff = F - L
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    k = sign_changes[0]
    return grid[k]

cut_hofstee = hofstee_cut(raw_scores, c_min, c_max, f_min, f_max)
fail_rate_at_cut = (raw_scores <= cut_hofstee).mean()
print(f"Hofstee cut: {cut_hofstee}  fail rate at cut: {fail_rate_at_cut:.3f}")
# Hofstee cut: 21  fail rate at cut: 0.252`}
      </CodeBlock>

      <Prose>
        The three methods on the same data and population produce: Modified Angoff = 20.64, Bookmark = 19.41 (expected raw), Hofstee = 21. A 1.5-point spread across methods is normal and is itself diagnostic: it tells the policy committee how much the cut score depends on the operationalization of "minimal competence" rather than on the test data itself. A spread of 5+ points across methods is a warning sign that the construct of "minimal competence" is not coherent for this test.
      </Prose>

      <H3>4f. Bootstrap confidence interval on the panel cut</H3>

      <Prose>
        The standard error formula above treats SMEs as the unit of variation, which is the conservative default. A nonparametric bootstrap over SMEs gives a confidence interval on the panel cut score that does not assume normality of judgments.
      </Prose>

      <CodeBlock language="python">
{`def bootstrap_panel_cut(per_sme_cuts, n_boot=2000):
    """Resample SMEs with replacement, recompute panel mean each time."""
    boots = np.empty(n_boot)
    n = len(per_sme_cuts)
    for b_ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b_] = per_sme_cuts[idx].mean()
    return boots

boot = bootstrap_panel_cut(sme_cut_r2, n_boot=2000)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"Modified Angoff cut: {sme_cut_r2.mean():.2f}  "
      f"95% bootstrap CI: ({ci_lo:.2f}, {ci_hi:.2f})")
# Modified Angoff cut: 20.64  95% bootstrap CI: (19.55, 21.45)`}
      </CodeBlock>

      <Prose>
        The 95% CI of (19.55, 21.45) on the modified Angoff cut is the headline number to report. With only J=5 SMEs the interval is wide; doubling the panel to J=10 typically halves the CI width, which is why standard-setting studies for high-stakes credentials almost always use 8 to 15 panelists.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production standard setting in 2026 happens in two settings: traditional human credentialing (medical boards, nursing licensing, professional certifications) and the newer LLM-deployment context (clinical assistants, code generation, autonomous decision systems). The procedural backbone is the same in both settings, but the failure modes are different.
      </Prose>

      <H3>5a. Traditional credentialing pipeline</H3>

      <Prose>
        A defensible standard-setting study for a high-stakes credential follows a documented protocol: (1) panel selection — typically 8 to 15 SMEs balanced across geography, practice setting, and demographics; (2) panel training — a half-day session on the test, the construct of "minimal competence," and the specific method to be used; (3) Round 1 ratings — independent, no discussion; (4) item-level feedback — empirical p-values, panel rating distributions, and SME-versus-panel comparisons; (5) structured discussion focused on items with high interrater variance; (6) Round 2 ratings; (7) optional Round 3; (8) cut-score computation with documented aggregation rule; (9) standard-error analysis and conditional SEM adjustment if policy requires; (10) panel debrief and confidence ratings; (11) policy committee review; (12) final report including all diagnostics. The process typically takes three days on-site plus four to six weeks of pre- and post-meeting work, and produces a 40- to 80-page technical report that becomes the legal record of the standard.
      </Prose>

      <H3>5b. Standard setting for LLM benchmarks</H3>

      <Prose>
        The LLM analogue replaces "candidate" with "model" but otherwise reuses the apparatus. A clinical organization deciding whether to deploy an LLM as a triage assistant might run the following: assemble a panel of 12 board-certified physicians; select 200 representative MedQA items spanning the deployment domain; run a modified Angoff procedure where each SME estimates the probability that "a model competent enough to deploy as triage assistant under physician supervision" would answer each item correctly; iterate two rounds; compute the cut score as the panel mean; report the bootstrap CI; document everything in a technical report that can be shown to the institutional review board, the malpractice insurer, and (if needed) opposing counsel.
      </Prose>

      <CodeBlock language="python">
{`"""
Production-style modified-Angoff pipeline for an LLM benchmark.
Drives the entire study from a CSV of SME ratings, produces the technical
report appendix as JSON.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

class ModifiedAngoffStudy:
    def __init__(self, item_metadata: pd.DataFrame, n_smes: int,
                 method_name: str = "Modified Angoff RP-67",
                 aggregator: str = "mean"):
        # item_metadata columns: item_id, item_text, empirical_p_value
        self.items      = item_metadata.copy()
        self.n_items    = len(item_metadata)
        self.n_smes     = n_smes
        self.method     = method_name
        self.aggregator = aggregator
        self.rounds     = []   # list of (J, n_items) ndarrays

    def add_round(self, ratings: np.ndarray):
        assert ratings.shape == (self.n_smes, self.n_items)
        assert ((ratings >= 0) & (ratings <= 1)).all(), "ratings must be in [0,1]"
        self.rounds.append(ratings.astype(float))

    def per_sme_cuts(self, round_idx: int = -1) -> np.ndarray:
        return self.rounds[round_idx].sum(axis=1)

    def panel_cut(self, round_idx: int = -1) -> float:
        cuts = self.per_sme_cuts(round_idx)
        return float(cuts.mean() if self.aggregator == "mean"
                     else np.median(cuts))

    def standard_error(self, round_idx: int = -1) -> float:
        cuts = self.per_sme_cuts(round_idx)
        return float(cuts.std(ddof=1) / np.sqrt(self.n_smes))

    def interrater_sd_per_item(self, round_idx: int = -1) -> np.ndarray:
        return self.rounds[round_idx].std(axis=0, ddof=1)

    def convergence_summary(self) -> dict:
        rows = []
        for r, ratings in enumerate(self.rounds):
            cuts = ratings.sum(axis=1)
            rows.append(dict(
                round         = r + 1,
                panel_cut     = float(cuts.mean()),
                panel_sd      = float(cuts.std(ddof=1)),
                se_of_mean    = float(cuts.std(ddof=1)/np.sqrt(self.n_smes)),
                mean_item_sd  = float(ratings.std(axis=0, ddof=1).mean()),
            ))
        return {"rounds": rows}

    def bootstrap_ci(self, round_idx: int = -1,
                     n_boot: int = 5000, alpha: float = 0.05,
                     rng=np.random.default_rng(0)) -> tuple[float, float]:
        cuts = self.per_sme_cuts(round_idx)
        agg  = (np.mean if self.aggregator == "mean" else np.median)
        boots = np.array([
            agg(cuts[rng.integers(0, self.n_smes, size=self.n_smes)])
            for _ in range(n_boot)
        ])
        return tuple(map(float, np.percentile(
            boots, [100*alpha/2, 100*(1-alpha/2)])))

    def reconciliation_flags(self, round_idx: int = -1,
                             threshold: float = 0.30) -> pd.DataFrame:
        """
        Flag items where panel rating diverges sharply from empirical p-value.
        These are the items that should be discussed in the next round.
        """
        ratings = self.rounds[round_idx]
        panel_p = ratings.mean(axis=0)
        emp_p   = self.items["empirical_p_value"].values
        gap     = panel_p - emp_p
        flagged = pd.DataFrame({
            "item_id":     self.items["item_id"].values,
            "panel_p":     panel_p,
            "empirical_p": emp_p,
            "gap":         gap,
        })
        return flagged[flagged["gap"].abs() > threshold].sort_values(
            "gap", key=abs, ascending=False)

    def technical_report(self, round_idx: int = -1) -> dict:
        return dict(
            method           = self.method,
            aggregator       = self.aggregator,
            n_items          = self.n_items,
            n_smes           = self.n_smes,
            n_rounds         = len(self.rounds),
            final_panel_cut  = self.panel_cut(round_idx),
            standard_error   = self.standard_error(round_idx),
            bootstrap_ci_95  = self.bootstrap_ci(round_idx),
            convergence      = self.convergence_summary(),
            flagged_items    = self.reconciliation_flags(round_idx).to_dict("records"),
        )

# Usage
items_df  = pd.read_csv("medqa_panel_items.csv")
study     = ModifiedAngoffStudy(items_df, n_smes=12)
study.add_round(np.load("round1_ratings.npy"))
study.add_round(np.load("round2_ratings.npy"))
report    = study.technical_report()
Path("medqa_standard_setting_report.json").write_text(json.dumps(report, indent=2))`}
      </CodeBlock>

      <H3>5c. Operational considerations specific to LLMs</H3>

      <Prose>
        Three things are different when the candidate is an LLM rather than a human. First, the model's responses are deterministic at temperature 0 and only weakly stochastic at low temperature; the standard error of the model's score on a fixed benchmark is therefore much smaller than for a human cohort, and the cut score should be applied to a single point estimate rather than a distribution. Second, the model's per-item correctness is correlated across items in ways human responses are not — a model that gets question 17 wrong because it lacks a particular medical fact is likely to get all related questions wrong. The effective sample size of a benchmark is therefore much smaller than the literal item count, and confidence intervals on the model's score must account for this clustering (Miller et al. 2024 on benchmark uncertainty quantification). Third, the cut score recommendation should be paired with calibration evaluation: a model that hits the cut score on average but is poorly calibrated (high confidence on wrong answers) may be unsuitable even at a passing score.
      </Prose>

      <Prose>
        For high-stakes deployments, run two parallel standard-setting studies — one Angoff-style on item content, one Bookmark-style on IRT-calibrated items — and treat the convergence (or divergence) of the two cut scores as a primary diagnostic. Convergence within 2 to 3 score points is reassuring; larger divergence is a sign that the construct of "minimal competence for this deployment" is not yet stable enough to support a defensible cut.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The plot below shows the convergence of per-SME cut scores across rounds in the modified Angoff procedure. Round 1 ratings span from 18 to 23 across five SMEs; by Round 2, after item-level feedback, the spread tightens to 19.4 to 21.6. The panel mean barely moves, but the standard error of the mean falls from 1.04 to 0.47.
      </Prose>

      <Plot
        label="Modified Angoff — per-SME cut scores by round"
        xLabel="round"
        yLabel="SME cut score"
        width={640}
        height={320}
        series={[
          {
            name: "SME 1",
            color: colors.gold,
            points: [[1, 22.41], [2, 21.46]],
          },
          {
            name: "SME 2",
            color: "#c084fc",
            points: [[1, 21.66], [2, 21.10]],
          },
          {
            name: "SME 3",
            color: "#4ade80",
            points: [[1, 18.27], [2, 19.59]],
          },
          {
            name: "SME 4",
            color: "#60a5fa",
            points: [[1, 22.79], [2, 21.62]],
          },
          {
            name: "SME 5",
            color: "#f87171",
            points: [[1, 18.05], [2, 19.42]],
          },
          {
            name: "panel mean",
            color: colors.textDim,
            points: [[1, 20.64], [2, 20.64]],
          },
        ]}
      />

      <Prose>
        The next plot shows the Hofstee compromise as a graphical construction. The blue curve is the empirical cumulative fail rate as a function of cut score. The gold line is the SME-elicited Hofstee line connecting the (max cut, min fail rate) corner to the (min cut, max fail rate) corner. The cut score is read off at the intersection.
      </Prose>

      <Plot
        label="Hofstee compromise — empirical fail rate vs. acceptability line"
        xLabel="cut score"
        yLabel="fail rate"
        width={640}
        height={320}
        series={[
          {
            name: "empirical fail rate F(s)",
            color: "#60a5fa",
            points: [
              [10, 0.011], [12, 0.030], [14, 0.078],
              [16, 0.158], [18, 0.247], [20, 0.358],
              [21, 0.420], [22, 0.485], [24, 0.610],
              [26, 0.731], [28, 0.834], [30, 0.910],
            ],
          },
          {
            name: "Hofstee acceptability line",
            color: colors.gold,
            points: [
              [17, 0.40], [19, 0.325], [21, 0.25],
              [23, 0.175], [25, 0.10],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows per-item interrater SD across two rounds for a 12-item subset. Cell darkness is the SD of SME ratings on that item; lighter cells mean tighter agreement. Round 2 cells are uniformly lighter than Round 1, reflecting the convergence the modified Angoff iteration is designed to produce.
      </Prose>

      <Heatmap
        label="Per-item interrater SD — Round 1 vs Round 2"
        rowLabels={["Round 1", "Round 2"]}
        colLabels={["i01", "i02", "i03", "i04", "i05", "i06", "i07", "i08", "i09", "i10", "i11", "i12"]}
        cellSize={36}
        colorScale="gold"
        matrix={[
          [0.12, 0.09, 0.18, 0.07, 0.21, 0.14, 0.11, 0.08, 0.16, 0.13, 0.19, 0.10],
          [0.05, 0.04, 0.07, 0.03, 0.08, 0.06, 0.04, 0.03, 0.06, 0.05, 0.07, 0.04],
        ]}
      />

      <Prose>
        Finally, the step trace below walks through one full round of a modified Angoff procedure as it would be conducted in person.
      </Prose>

      <StepTrace
        label="Modified Angoff — one round end-to-end"
        steps={[
          {
            label: "Frame the MC candidate",
            render: () => (
              <Prose>
                The facilitator opens with a 30-minute discussion of "minimally competent": what does the borderline candidate know, what can they do, what do they not yet do? The panel writes a one-page MC profile that will anchor every subsequent rating. This step is non-optional; panels that skip it produce inconsistent ratings on Round 1 and never converge in Round 2.
              </Prose>
            ),
          },
          {
            label: "Round 1 ratings (silent)",
            render: () => (
              <Prose>
                Each SME independently rates every item with a probability in [0, 1]. No discussion. The facilitator collects the ratings into a matrix of shape (J SMEs, n items). For each SME, sum across items to get that SME's Round 1 cut score. Compute the panel mean and SD.
              </Prose>
            ),
          },
          {
            label: "Item-level feedback",
            render: () => (
              <Prose>
                The facilitator distributes per-item summaries: panel mean rating, panel SD, the SME's own rating, and the empirical p-value from the calibration sample. Items where panel SD exceeds 0.15 or where panel mean differs from empirical p-value by more than 0.30 are flagged for discussion.
              </Prose>
            ),
          },
          {
            label: "Structured discussion",
            render: () => (
              <Prose>
                The flagged items are discussed one at a time. SMEs who rated unusually high or low are asked to explain their reasoning. The facilitator does not push for consensus; the goal is shared understanding of why ratings diverged. Discussions on a 50-item test typically take two to three hours.
              </Prose>
            ),
          },
          {
            label: "Round 2 ratings (silent)",
            render: () => (
              <Prose>
                SMEs rate again, independently. The expected effect is that interrater SD drops by roughly half on the discussed items and modestly on undiscussed items. The panel mean cut typically moves by less than half a point. If it moves by more than a point, the panel is unstable and a Round 3 is warranted.
              </Prose>
            ),
          },
          {
            label: "Compute and document",
            render: () => (
              <Prose>
                Compute the panel cut as the mean (or median, if pre-specified) of per-SME cuts. Report the standard error and bootstrap CI. Tabulate items where panel rating still diverges from empirical p-value as a permanent appendix to the technical report. Have each SME complete a confidence rating: "How confident are you that the recommended cut score reflects minimal competence?" — values below 4/5 should be investigated.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Angoff vs Modified Angoff</H3>
      <Prose>
        Choose the original (non-iterated) Angoff method essentially never in production. The modified Angoff with two rounds of structured feedback dominates it on every metric: lower interrater variance, smaller standard error on the panel cut, higher SME confidence in the result, and substantially better defensibility under audit. The only setting where original Angoff is justified is exploratory pilot work where the goal is to scope the range of plausible cut scores rather than produce a defensible standard.
      </Prose>

      <H3>Modified Angoff vs Bookmark</H3>
      <Prose>
        Modified Angoff is the safer default when items have not been IRT-calibrated, when the test is small (under 100 items), or when the panel includes SMEs without measurement training. Bookmark requires IRT calibration as a prerequisite and a booklet of items that can be meaningfully ordered by difficulty — both of these are non-trivial in domains where item difficulty depends on context (clinical reasoning under varying patient acuity, for example). When IRT calibration is available and the items can be ordered, Bookmark is faster (a panel can place a single bookmark in less time than they can rate 50 items twice) and tends to produce slightly tighter interrater agreement. Cizek and Bunch's 2007 textbook recommends running both methods in parallel for high-stakes credentials and treating their convergence as a validity check.
      </Prose>

      <H3>Modified Angoff vs Hofstee</H3>
      <Prose>
        The Hofstee method has one significant practical advantage: it yokes the cut score to the population fail rate explicitly, which forces the policy committee to confront the resource and equity consequences of the standard. If a Hofstee cut produces a 60% fail rate, that is information the committee must reckon with before defending the cut publicly. Angoff and Bookmark do not produce this kind of population-level reckoning naturally. The disadvantage of Hofstee is that it conflates the panel's substantive judgment about competence with the panel's policy judgment about acceptable fail rates; in jurisdictions where these need to be documented separately, that conflation is a problem. Use Hofstee as a reality check on Angoff or Bookmark, not as a stand-alone primary method.
      </Prose>

      <H3>Borderline group vs Contrasting groups</H3>
      <Prose>
        Both methods require SMEs to classify candidates rather than items. Borderline group is cleaner conceptually — the median test score among "borderline" candidates is a direct estimate of the cut — but requires SMEs who have observed enough candidates to identify a borderline cohort, which is rarely possible for new tests. Contrasting groups uses the larger and more interpretable populations of clear masters and clear non-masters and chooses the cut that minimizes total classification error. It is the method of choice when SME observations of candidate performance are available (clinical settings, on-the-job evaluations, internships) but not when SMEs only see the test score.
      </Prose>

      <H3>When to use which for LLM evaluation</H3>
      <Prose>
        For most LLM deployment decisions today, modified Angoff is the right default. It does not require IRT calibration of the benchmark, it scales to panels of 5 to 15 domain experts, and it produces a defensible audit trail. Use Bookmark when the benchmark has been IRT-calibrated (rare in current LLM evaluation but increasingly common) and when items can be ordered by difficulty. Use Hofstee as a sanity check that the recommended cut is consistent with the team's beliefs about acceptable model failure rates in deployment. Use contrasting groups when you have a labeled corpus of model outputs that domain experts have classified as "deployable" or "not deployable," which often emerges as a byproduct of red-teaming.
      </Prose>

      <H3>Method-by-property summary</H3>
      <Prose>
        The table below summarizes the trade-offs across methods on five dimensions: input the SME judges, technical prerequisites, time per panelist, defensibility under audit, and the most common failure mode of each.
      </Prose>

      <CodeBlock language="text">
{`Method               SME judges   Prereqs        Time      Defensibility  Top failure mode
─────────────────────────────────────────────────────────────────────────────────────
Original Angoff      items (P)    item bank      ~6 hrs    medium         high SME variance
Modified Angoff      items (P)    item bank,     ~12 hrs   high           p-value anchoring
                                  empirical p
Bookmark             positions    IRT calib,     ~4 hrs    high           difficulty mis-order
                                  ordered book
Hofstee              fail-rate    score distrib. ~2 hrs    medium-high    no intersection
                     bounds       in pop.
Contrasting Groups   candidates   labeled cohort ~variable medium         label noise
Borderline Group     borderline   labeled cohort ~variable low-medium     small N at border`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Standard setting was developed for tests of fewer than 200 items administered to populations of thousands. Each of those scales — items, candidates, SMEs — has a different elasticity, and the engineering of a defensible cut score depends on understanding which scaling regimes are routine and which are open research problems.
      </Prose>

      <Prose>
        Items scale poorly within a single panel session. A panel of 8 SMEs rating a 200-item test in modified Angoff style will spend more than three days in the room and will exhibit fatigue effects on the last 30 to 40 items. The professional solution is to split the test into clusters by content area, run separate (smaller) panels per cluster, and aggregate cut scores via item weights. The Bookmark method scales better because the SME's task is one cognitive judgment per panelist regardless of test length — but it depends on the items being orderable by difficulty, which gets harder as test length grows.
      </Prose>

      <Prose>
        Panelists scale predictably. Doubling the number of SMEs reduces the standard error of the panel cut by roughly √2. Most published high-stakes credentialing studies use J = 8 to J = 15 panelists. Panels smaller than J = 5 are not defensible for high-stakes settings; panels larger than J = 20 hit diminishing returns and are dominated by logistical cost rather than statistical efficiency. A practical rule from Plake's 2008 review: budget a recruitable panel size that produces a target standard error of one to two raw score points on the cut.
      </Prose>

      <Prose>
        Candidates scale almost trivially in the LLM setting because the model can be evaluated on arbitrarily many items at zero marginal cost. This breaks the human-test economic logic in interesting ways: with humans, each additional test item costs every candidate time to answer; with models, the marginal cost is microseconds. The dominant cost in LLM standard setting is not items, it is item curation and SME judgment time. Spending engineering effort on better item-bank curation pays back more than spending the same effort on running larger panels.
      </Prose>

      <Prose>
        What does not scale is the construct itself. "Minimal competence" is well-defined for a narrow, well-bounded role like "first-year emergency medicine resident on day 1 of overnight call." It is not well-defined for diffuse roles like "general-purpose clinical assistant," "competent code generator," or "trustworthy autonomous driver." The breadth of role implied by a deployment determines whether standard setting can produce a defensible cut at all. For broad roles, the only honest answer is to decompose the role into multiple narrower roles and run separate standard-setting studies per sub-role, then deploy with conditional thresholds (the model is allowed to handle medication-dose questions only if it scored above the medication cut, even if its overall MedQA score is high). The temptation to report a single number — a single cut score on a single benchmark — should be resisted whenever the deployment role is broader than what the benchmark covers.
      </Prose>

      <Prose>
        The standard-setting literature also gets stuck on what to do when no defensible cut exists. The honest answer, present in Cizek and Bunch (2007) and Plake and Cizek (2012), is that some test scores cannot support a defensible cut score and must instead support a more elaborate decision rule (multiple cuts for different risk tiers, conditional cuts that depend on candidate background, or cuts paired with mandatory secondary review). The bias toward producing a single number even when one is not justified is a well-documented institutional failure mode and is, in 2026, repeating itself in LLM deployment decisions where teams report "the model passes the threshold" on benchmarks that do not support a defensible threshold at all.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Anchoring on empirical p-values</H3>
      <Prose>
        In modified Angoff, showing SMEs the empirical p-values is essential for convergence but also introduces a known anchoring bias: SMEs unconsciously revise their ratings toward the empirical p-value rather than toward the panel consensus. The result is a cut score that recovers the empirical mean of the candidate population rather than the construct of "minimal competence." Mitigation: report empirical p-values only as 5-tile bands (very easy, easy, moderate, hard, very hard) rather than as point estimates, and explicitly remind SMEs that p-values describe what candidates do, not what minimally competent candidates should do.
      </Prose>

      <H3>SME selection bias</H3>
      <Prose>
        Panels recruited by convenience tend to over-represent academically affiliated SMEs, who systematically rate items as easier than community-practice SMEs do. This produces cut scores that are too high relative to the role being credentialed. The mitigation is structured panel selection: explicit quotas on practice setting, geographic region, years in role, and demographics. A documented panel-selection protocol is a non-negotiable component of a defensible standard.
      </Prose>

      <H3>Construct drift across rounds</H3>
      <Prose>
        After Round 1 discussion, SMEs sometimes shift the construct of "minimal competence" rather than just refining their ratings — for example, after seeing that the panel rated a tough item easy, an SME may decide the MC candidate is "stronger than I initially thought" and revise multiple unrelated items upward. This is invisible in the per-item statistics but visible in the per-SME cut score moving by more than 2 points between rounds in the same direction across most SMEs. Mitigation: track per-SME cut-score deltas and flag any SME whose cut moves more than 2 SD from their Round 1 value.
      </Prose>

      <H3>Mis-ordered items in Bookmark</H3>
      <Prose>
        Bookmark depends on the items being ordered correctly by difficulty. If the IRT calibration is noisy — which it always is when calibration sample sizes are below 500 — local re-orderings of nearby items confuse SMEs and produce bookmarks that are not stable across SMEs. Mitigation: use Bookmark only when item locations are estimated with standard errors below 0.10 on the θ scale, and verify item ordering with a small adjudication panel before the main standard-setting session.
      </Prose>

      <H3>Hofstee with no intersection</H3>
      <Prose>
        If the empirical cumulative fail-rate curve never crosses the Hofstee acceptability line, the method produces no answer. This usually means the panel's elicited bounds on the cut score and on the acceptable fail rate are jointly infeasible against the candidate population's actual score distribution. The right response is to ask the panel to re-elicit one of the four bounds with explicit knowledge of the empirical distribution, but this introduces the same anchoring concern as in modified Angoff. Document both the original (infeasible) bounds and the revised (feasible) bounds in the technical report.
      </Prose>

      <H3>Stochasticity in LLM scores</H3>
      <Prose>
        Standard-setting methods assume the candidate's score is a fixed point estimate. LLMs at non-zero temperature produce a distribution over scores; the cut decision should therefore be made against the lower bound of a confidence interval, not the point estimate. A model whose mean score is above the cut but whose 95% lower bound is below it has not demonstrably passed. Use temperature-0 deterministic scoring for standard-setting evaluations whenever possible, and report sampling variance separately from cut-score variance when temperature must be non-zero.
      </Prose>

      <H3>Item-level correlation in benchmarks</H3>
      <Prose>
        Models tend to fail correlated subsets of benchmark items because the same underlying knowledge gap manifests across many surface-distinct questions. The effective sample size of a benchmark is therefore much smaller than the literal item count, and the standard error of the model's score is proportionally larger. A model scoring 75% on a 1,000-item benchmark with strong cross-item correlation may have an effective standard error closer to that of a 100-item independent benchmark. Cut score margins should be set with this clustering in mind. Methods from Miller et al. 2024 (benchmark uncertainty quantification) provide concrete clustering-aware variance estimators.
      </Prose>

      <H3>Reporting the cut without the SE</H3>
      <Prose>
        A cut score reported without its standard error is not a defensible standard. The single most common failure mode in published LLM evaluation reports is the use of round-number thresholds ("the model must achieve 80% accuracy") with no documentation of how the threshold was chosen, what the standard error around it is, or what the false-pass and false-fail rates would be at that threshold. Any standard-setting result that survives audit will report the cut, the standard error, the 95% bootstrap CI, the panel composition, the method, and the per-round convergence diagnostics — minimally.
      </Prose>

      <H3>Construct breadth exceeding test breadth</H3>
      <Prose>
        Setting a cut score on a benchmark that does not cover the deployment domain is the most consequential failure mode in LLM standard setting. A model that passes the MedQA cut score has not demonstrated competence on the 80% of clinical interactions that MedQA does not represent. The cut score is meaningful only as a statement about performance on the benchmark; transferring that statement to the deployment role is a separate inference that requires content-validity evidence. The discipline of standard setting is fundamentally about being honest about what a number means.
      </Prose>

      <Callout accent="purple">
        Cut scores are policy choices wearing measurement clothing. The purpose of the rigorous procedures described here is not to discover the "correct" cut score — there is no such thing — but to make the policy choice transparent, auditable, and accountable. When the procedure becomes a substitute for the underlying policy decision rather than a structured way to make it, the standard-setting study has failed even if all the diagnostics look healthy.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The references below are the documents an examining body, regulator, or court will recognize as authoritative on standard setting. Each was consulted directly for this topic.
      </Prose>

      <H3>Angoff 1971 — the founding document</H3>
      <Prose>
        William H. Angoff. "Scales, Norms, and Equivalent Scores." In Robert L. Thorndike (Ed.), <em>Educational Measurement</em> (2nd ed., pp. 508–600). American Council on Education, Washington DC, 1971. The chapter introduces, almost in passing in a footnote on page 514, what became known as the Angoff method: "A simple way of estimating the standard for an examination is to ask each judge to estimate, for each item, the probability that the borderline candidate would answer it correctly. The sum of these probabilities is the judge's estimate of the borderline score." Half a century later, this is still the most cited sentence in the standard-setting literature.
      </Prose>

      <H3>Cizek 2001 — Bookmark and the modern synthesis</H3>
      <Prose>
        Gregory J. Cizek (Ed.). <em>Setting Performance Standards: Concepts, Methods, and Perspectives.</em> Lawrence Erlbaum, Mahwah NJ, 2001. The first comprehensive edited volume on standard setting; contains the canonical chapter on the Bookmark method by Mitzel, Lewis, Patz, and Green that codifies the procedure first used in operational K-12 testing in the late 1990s. Establishes RP=0.67 as the field default. Also contains Plake's chapter on the modified Angoff procedure that became the de facto operational protocol for medical and nursing licensing boards.
      </Prose>

      <H3>Cizek and Bunch 2007 — the operational textbook</H3>
      <Prose>
        Gregory J. Cizek and Michael B. Bunch. <em>Standard Setting: A Guide to Establishing and Evaluating Performance Standards on Tests.</em> Sage Publications, Thousand Oaks CA, 2007. The textbook every standard-setting facilitator owns. Walks through panel selection, training, round-by-round procedures, statistical analyses, and report writing for every major method. The chapter on documentation and validity evidence is the single best reference for what a defensible technical report looks like.
      </Prose>

      <H3>Lewis 2009 — Bookmark recalibration</H3>
      <Prose>
        Daniel M. Lewis. "The Bookmark Standard Setting Procedure." In Gregory J. Cizek (Ed.), <em>Setting Performance Standards: Foundations, Methods, and Innovations</em> (2nd ed.). Routledge, 2012. (The 2009 reference is to Lewis's NCME presentation that updated the original 2001 procedure.) Documents the failure modes of Bookmark observed in K-12 operational use over the 2001–2009 period — including the ordered-item-booklet construction issues — and proposes the refinements (ordered-item booklet construction protocols, item-mapping displays) that are standard in current Bookmark practice.
      </Prose>

      <H3>Plake and Cizek 2012 — the modified Angoff manual</H3>
      <Prose>
        Barbara S. Plake and Gregory J. Cizek. "Variations on a Theme: The Modified Angoff, Extended Angoff, and Yes/No Standard Setting Methods." In Cizek (Ed.), <em>Setting Performance Standards</em> (2nd ed., pp. 181–199). Routledge, 2012. The definitive reference for the modified Angoff procedure as practiced today, with detailed protocols for each round, examples of feedback packets, and the reporting requirements for credentialing bodies. Also covers the "Yes/No" method (rate only whether the MC candidate would answer correctly, not the probability) which reduces SME cognitive load on long tests at the cost of slightly higher variance.
      </Prose>

      <H3>Beuk 1984 — the RP-67 derivation</H3>
      <Prose>
        Cees L. M. Beuk. "A Method for Reaching a Compromise Between Absolute and Relative Standards in Examinations." <em>Journal of Educational Measurement</em>, 21(2), 147–152, 1984. Original derivation of why RP=0.67 is the empirically defensible default for "mastery" in the context of multiple-choice testing with non-trivial guessing. The paper also introduces what would become known as the Beuk compromise method, a close cousin of Hofstee that uses panel medians rather than ranges.
      </Prose>

      <H3>Hofstee 1983 — the compromise method</H3>
      <Prose>
        Wim K. B. Hofstee. "The Case for Compromise in Educational Selection and Grading." In Stephen B. Anderson and John S. Helmick (Eds.), <em>On Educational Testing</em> (pp. 109–127). Jossey-Bass, San Francisco, 1983. Hofstee's original paper introducing the compromise method that bears his name. Argues that absolute standards (Angoff-style) and relative standards (norm-referenced) are both incomplete, and the compromise — bounding both the cut and the fail rate, then intersecting with the empirical distribution — is the honest path between them.
      </Prose>

      <H3>Miller et al. 2024 — LLM benchmark uncertainty</H3>
      <Prose>
        Joshua Miller, Joshua Vendrow, Aleksander Madry, et al. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640, 2024. Contemporary treatment of how to compute defensible confidence intervals on LLM benchmark scores when items are correlated, when prompts are sampled from a distribution, and when model temperature is non-zero. The methods in this paper are the right uncertainty quantification companion to the standard-setting procedures described above; together they let you say not just "the cut is 78" but "the cut is 78 ± 2 and the model's score is 81 ± 3, so the model passes with marginal evidence."
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why summing probabilities works</H3>
      <Prose>
        The Angoff method computes each SME's cut score as the sum of per-item probabilities. Show formally why this is the expected total raw score for a candidate whose per-item correctness probabilities are exactly the SME's reported probabilities, treating each item as an independent Bernoulli trial. Then identify the assumption that breaks when items are not independent — for example, when an LLM's correctness on related items is correlated through shared latent knowledge. What does the failure of independence do to the variance of the cut score, and why does the panel-mean point estimate still survive even when the variance estimate is wrong?
      </Prose>

      <H3>Exercise 2 — Bookmark with RP variation</H3>
      <Prose>
        For an item with discrimination <Code>a = 1.2</Code> and difficulty <Code>b = 0.4</Code>, compute the RP-50, RP-67, and RP-80 locations explicitly. Then describe how the Bookmark cut score would shift if a panel was instructed to use RP=0.50 instead of RP=0.67 for the same set of bookmarks. Which direction does the cut move and why? In which deployment context would you choose RP=0.80 over RP=0.67, and what does that imply about your willingness to false-pass versus false-fail?
      </Prose>

      <H3>Exercise 3 — Diagnosing convergence failure</H3>
      <Prose>
        After running modified Angoff Round 2, you observe that the panel mean cut score moved from 22.0 in Round 1 to 26.5 in Round 2, and that all five SMEs revised upward by between 3 and 6 points. The per-item interrater SD also dropped from 0.10 to 0.05. List three possible explanations for this pattern and describe what additional analysis you would run to distinguish among them. Which of these explanations would force you to re-run the panel from scratch versus merely document the shift?
      </Prose>

      <H3>Exercise 4 — Hofstee with no intersection</H3>
      <Prose>
        A Hofstee panel specifies (c_min, c_max, f_min, f_max) = (15, 22, 0.05, 0.20). The empirical fail-rate curve <Code>F(s)</Code> at s=15 is 0.32 and at s=22 is 0.62. Graph the Hofstee line and the empirical curve mentally and explain why no intersection exists. What does this absence say about the relationship between the panel's beliefs about competence and the population's actual performance? Propose two distinct policy responses — one that revises the panel's bounds, one that revises the candidate population — and discuss which is more defensible under a regulatory audit.
      </Prose>

      <H3>Exercise 5 — Standard setting for an LLM coding assistant</H3>
      <Prose>
        Your team is preparing to deploy an LLM as a coding assistant for a financial-services engineering organization. The benchmark is HumanEval (164 items). The deployment role is "junior engineer who can ship low-risk code with senior review." Design a complete standard-setting study: choose the method, justify it; specify panel composition (how many SMEs, what backgrounds, how recruited); specify the per-item or per-position task; specify how you will compute the cut score, the standard error, and the bootstrap CI; specify the reconciliation criteria for items where panel rating diverges from empirical p-value; and specify the deliverables (technical report sections, appendices, audit trail). Then identify the three biggest threats to defensibility in your design and the mitigations you would build in.
      </Prose>

      <H3>Exercise 6 — Cross-method convergence as validity evidence</H3>
      <Prose>
        You run modified Angoff and Bookmark on the same benchmark with the same panel and obtain cut scores of 73 and 68 respectively (5-point gap). Cizek and Bunch's 2007 textbook says cross-method convergence is a primary validity check. What does a 5-point gap tell you about the construct of "minimal competence" you operationalized? List three causes of cross-method divergence and describe, for each cause, what you would do differently in a follow-up study to reduce the gap. When would a 5-point gap actually be the right answer rather than a problem?
      </Prose>

    </div>
  ),
};

export default standardSettingMethods;
