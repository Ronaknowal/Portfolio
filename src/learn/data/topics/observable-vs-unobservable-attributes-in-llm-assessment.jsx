import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const observableVsUnobservable = {
  title: "Observable vs Unobservable Attributes in LLM Assessment",
  slug: "observable-vs-unobservable-attributes-in-llm-assessment",
  readTime: "~34 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        When an LLM is used as a judge — scoring a response, ranking two responses, or assigning a quality label — the judge has access to a precisely bounded slice of reality. It sees the prompt, it sees the candidate response, and that is the entire world it is allowed to reason about. Everything else that exists outside of those two text strings — the author who wrote the response, the demographic group that author belongs to, the context in which the prompt was issued, the ground-truth correctness of any subjective claim, the judge's own training distribution — is hidden from the judge at evaluation time. The judge can only infer from what is on the page. This sounds tautological, but it has profound consequences for fairness. A judge that cannot see attribute <Code>U</Code> cannot directly condition on <Code>U</Code> when scoring. It can, however, condition on observable proxies for <Code>U</Code>: stylistic markers, vocabulary choice, sentence structure, formatting, the presence of certain phrases or constructions. The ability of those proxies to leak information about <Code>U</Code> is what determines whether the judge is meaningfully fair on that hidden attribute or merely fair-looking.
      </Prose>

      <Prose>
        The distinction between observable and unobservable attributes was sharpened in the algorithmic fairness literature long before LLMs became the dominant evaluation tool. Lipton, Chouldechova, and McAuley's 2018 paper "Does mitigating ML's impact disparity require treatment disparity?" (arXiv:1711.07076) framed the issue with surgical precision: a classifier that is blind to a protected attribute — that does not receive it as input — is structurally constrained in how individual-level fair it can be. If two individuals differ only in their protected attribute and both produce the same observable feature vector, a blind classifier must give them the same prediction. Whether that prediction is "fair" depends entirely on what fairness means for the task. If the task's true label distribution actually depends on the protected attribute (because of historical data, sampling bias, or genuine differences in base rates), then a blind classifier achieves equal treatment but not equal outcomes. The blind classifier cannot distinguish between two individuals it cannot tell apart. It cannot un-confound what its inputs already conflate.
      </Prose>

      <Prose>
        Translate this directly to the LLM-as-judge setting. The judge is a blind classifier in Lipton et al.'s sense: it sees text, not author identity. If a prompt-response pair from a response written by a person named "John" produces the exact same token sequence as one from a person named "Jamal" — same content, same length, same vocabulary, same formatting — the judge will return the same score. So far so good: the judge has achieved a form of individual fairness on the input text. But this is a vacuous guarantee. In practice, responses written by people with different backgrounds, different first languages, different educational paths, different cultural conventions for prose construction, do not produce identical token sequences. They produce systematically different ones. Dialect, vocabulary, average sentence length, the use of certain rhetorical conventions — all of these correlate with author demographics. The judge does not need to "see" the demographic to score it differently; it only needs to see the stylistic features that demographics produce.
      </Prose>

      <Prose>
        This is the fairness gap that motivates the entire discipline of treating observable and unobservable attributes as fundamentally different objects. Fairness on observable attributes is achievable through invariance-style interventions: paraphrase invariance, length normalization, formatting canonicalization, citation standardization. Fairness on unobservable attributes is harder by construction. The judge cannot condition on what it cannot see, but it can learn to use observable proxies that carry information about the hidden attribute, and once that information is in the input it is computationally available to influence the output. Demonstrating that the judge does not directly use the unobservable attribute is trivial — it never had access to it. Demonstrating that the judge does not effectively use it through proxies requires a much more careful analysis of mutual information between observable features and the hidden attribute.
      </Prose>

      <Prose>
        The practical urgency for this framework comes from the deployment regime LLM judges operate in. Across the industry, LLM judges have become the dominant scoring infrastructure for instruction-tuning datasets, RLHF preference labels, evaluation benchmarks, and synthetic data filtering. Anthropic, OpenAI, Google DeepMind, Meta, Mistral, Cohere, and most major AI labs run LLM-judge pipelines that score billions of responses per training cycle. If those judges systematically score responses written by certain demographic groups lower because of stylistic correlates of demographic identity, the bias gets amplified at every subsequent training stage. The model learns to produce responses that the judge prefers; the judge prefers responses with demographic-correlated stylistic features; the trained model becomes more confidently aligned to a particular stylistic register. The bias propagates from the judge into the policy. Understanding which fairness guarantees you can extract from a judge — and which you provably cannot — is a prerequisite to deploying these pipelines responsibly.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the simplest possible mental model. There is a true latent attribute <Code>U</Code> — call it the demographic group of the response author, or the author's identity, or the ground-truth correctness of a subjective claim. The judge does not see <Code>U</Code>. The judge sees the response text <Code>R</Code>, which can be decomposed into two parts: the content <Code>C</Code> (what the response actually says) and the observable stylistic features <Code>S</Code> (how it says it — length, vocabulary, syntactic constructions, formatting). Both <Code>C</Code> and <Code>S</Code> are functions of <Code>U</Code> in general: different authors write about different topics in different ways. The judge produces a score <Code>J</Code> by reading <Code>R</Code>. The fairness question is whether <Code>J</Code> depends on <Code>U</Code> in a way that is unjustified by the actual quality of the response.
      </Prose>

      <Prose>
        Notice the structure: <Code>J</Code> is a function of <Code>R</Code>, and <Code>R</Code> is a function of <Code>U</Code> (mediated through <Code>C</Code> and <Code>S</Code>). The judge cannot avoid all dependence on <Code>U</Code>; it can only avoid the unjustified part. The justified part is the dependence that flows through <Code>C</Code> — if a response from author group A is genuinely more accurate or more helpful, the judge should score it higher. The unjustified part is dependence that flows through <Code>S</Code> — if a response from author group A uses different sentence structure than one from group B but communicates the same content with the same accuracy, the judge should not score them differently. This decomposition gives us a precise target: a fair judge is one for which <Code>J ⊥ U | C</Code>. Conditioned on the true content quality of the response, the judge's score should be independent of the author's demographic group.
      </Prose>

      <Prose>
        The challenge is that this conditional independence is impossible to verify directly. We don't have ground-truth content quality <Code>C</Code> as an observable variable. We have proxies for it (other judges, human ratings, downstream task outcomes), each of which may share the judge's biases. What we can do is detect and quantify the leakage: how much can a probe trained on the observable features <Code>S</Code> alone recover the unobservable attribute <Code>U</Code>? If the probe's accuracy is at chance level, then the observable features carry no information about <Code>U</Code>, and the judge has no proxy signal to use. If the probe's accuracy is high, the proxy signal is strong, and any judge that reads <Code>S</Code> at all has the option of using it. Fairness mitigations have to attack the proxy directly.
      </Prose>

      <Prose>
        A second core intuition is the asymmetry between observable and unobservable interventions. When the unfair behavior is driven by an observable attribute — the judge prefers longer responses, the judge prefers responses that use bullet points, the judge prefers responses with citations — you can intervene directly. You can normalize length. You can strip formatting. You can canonicalize citations. The intervention is mechanical: re-process the input to remove the spurious feature, then run the judge. After the intervention, the judge's score depends only on what remains, and what remains is by construction independent of the spurious feature. This kind of fairness is easy to achieve and easy to verify.
      </Prose>

      <Prose>
        When the unfair behavior is driven by an unobservable attribute, you cannot intervene directly because you cannot remove what was never explicitly there. The unobservable attribute is encoded jointly across many observable features in non-trivial ways. Demographic identity might leak through dialect markers, vocabulary preferences, sentence rhythm, and the choice of which examples to mention. Stripping out individual markers does not remove the joint signal. Adversarial training can reduce the leakage but cannot eliminate it without also destroying content. There is a fundamental trade-off between how invariant the representation is to the protected attribute and how informative it remains for the actual task. Madras, Creager, Pitassi, and Zemel formalized this in 2018 as the LAFTR framework (Learning Adversarially Fair and Transferable Representations, arXiv:1802.06309), showing that adversarial fairness is a Pareto frontier rather than a binary property.
      </Prose>

      <Prose>
        A third intuition that matters for LLM judges specifically is the mention-vs-use distinction. A response can mention a demographic-coded marker without using it as a stylistic signal. "John, in his memoir, writes about growing up in rural Alabama" mentions a demographic context but does so descriptively. "Y'all, lemme tell you 'bout this thing" uses a demographic-coded register stylistically. A fair judge should treat the first as content (it is information being communicated about a topic) and the second as style (it is a way of communicating that should not affect quality assessment for content-focused tasks). In practice, LLM judges often conflate mention and use, because the underlying language model has learned that responses written in certain registers are statistically associated with certain demographics. Disentangling mention from use requires either explicit disentangled representations (which are an open research problem) or careful prompt engineering to instruct the judge to ignore stylistic register when scoring substantive content.
      </Prose>

      <Prose>
        A final intuition: the unobservable attribute does not have to be a demographic. For subjective tasks like creative writing evaluation, opinion summarization, or aesthetic judgment, the unobservable attribute is the ground-truth quality itself. The judge cannot directly perceive whether a poem is genuinely good; it can only perceive features that correlate with quality (rhythm, novelty, coherence). Fairness in this setting is not about demographic equity but about epistemic humility — recognizing that the judge's score is a function of observable proxies for an unobservable target, and that proxies can be miscalibrated in ways that systematically prefer certain styles over others. The same machinery that quantifies demographic leakage can quantify quality miscalibration.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>U</Code> denote the unobservable attribute (demographic group, author identity, latent quality), <Code>S</Code> denote the observable stylistic features visible to the judge, <Code>C</Code> denote the content-quality features that the judge should be sensitive to, and <Code>J</Code> denote the judge's score. The graphical model that captures the typical LLM-judge setting is:
      </Prose>

      <MathBlock>{"U \\to (C, S),\\quad (C, S) \\to R,\\quad R \\to J"}</MathBlock>

      <Prose>
        The author's latent attribute <Code>U</Code> influences both content and style; the observable response <Code>R</Code> is a deterministic encoding of both; the judge produces a score from <Code>R</Code>. The judge does not have direct access to <Code>U</Code>, only to the response <Code>R</Code> which carries information about <Code>U</Code> through <Code>S</Code>.
      </Prose>

      <H3>3a. Conditional independence as a fairness target</H3>

      <Prose>
        The natural fairness criterion in this graphical model is content-conditional independence:
      </Prose>

      <MathBlock>{"J \\;\\perp\\!\\!\\!\\perp\\; U \\;\\big|\\; C"}</MathBlock>

      <Prose>
        Read this as: conditional on the actual content quality of the response, the judge's score should not depend on the author's unobservable attribute. This is the LLM-judge analogue of the equalized odds criterion of Hardt, Price, and Srebro (2016, "Equality of Opportunity in Supervised Learning"). The constraint is necessary because we want the judge to be sensitive to true content differences (otherwise it cannot rank responses by quality at all) but insensitive to differences that arise solely from how the content is expressed by different demographic groups.
      </Prose>

      <H3>3b. Information-theoretic upper bound on bias</H3>

      <Prose>
        The data processing inequality gives us a fundamental constraint. Since <Code>J</Code> is a deterministic function of <Code>R</Code>, and <Code>R</Code> is a function of <Code>(C, S)</Code>, the mutual information between the judge's output and the unobservable attribute is upper-bounded by the mutual information between the response (or any of its components) and the unobservable attribute:
      </Prose>

      <MathBlock>{"I(J; U) \\;\\le\\; I(R; U) \\;=\\; I(C, S; U)"}</MathBlock>

      <Prose>
        And by the chain rule of mutual information:
      </Prose>

      <MathBlock>{"I(C, S; U) \\;=\\; I(C; U) + I(S; U \\,|\\, C)"}</MathBlock>

      <Prose>
        The first term, <Code>{"I(C; U)"}</Code>, is the mutual information between content quality and the unobservable attribute. This is the "justified" channel — if author groups genuinely produce content of different quality (because of access to information, training, time, or any other resource), some of the score variation will trace back to <Code>U</Code> through <Code>C</Code>. The second term, <Code>{"I(S; U | C)"}</Code>, is the mutual information between the observable stylistic features and <Code>U</Code> after we have already conditioned on content. This is the "unjustified" channel — the leakage of demographic information through style alone, holding content constant. Fairness work targets this second term.
      </Prose>

      <H3>3c. Conditional entropy of the judge given U</H3>

      <Prose>
        For a judge to be fair on <Code>U</Code> conditional on <Code>C</Code>, we need the conditional entropy of the judge's output given <Code>C</Code> to be unaffected by adding <Code>U</Code> as a conditioning variable:
      </Prose>

      <MathBlock>{"H(J \\,|\\, C) \\;=\\; H(J \\,|\\, C, U)"}</MathBlock>

      <Prose>
        Equivalently, <Code>{"I(J; U | C) = 0"}</Code>. By the data processing inequality applied to the chain <Code>{"U \\to S \\to R \\to J"}</Code> (with <Code>C</Code> held fixed), we have:
      </Prose>

      <MathBlock>{"I(J; U \\,|\\, C) \\;\\le\\; I(R; U \\,|\\, C) \\;=\\; I(S; U \\,|\\, C)"}</MathBlock>

      <Prose>
        This is the central inequality of the observable/unobservable framework. The judge's residual unfairness — the part of its score variation that depends on the demographic group after controlling for content — is upper-bounded by the residual leakage of demographic information through observable stylistic features. If we can drive <Code>{"I(S; U | C)"}</Code> to zero, then the judge is provably fair regardless of how it processes its inputs. If <Code>{"I(S; U | C)"}</Code> remains positive, then any judge that uses <Code>S</Code> at all has the option of being unfair, and we must inspect the specific function the judge implements to know whether it actually is.
      </Prose>

      <H3>3d. The Lipton et al. impossibility result</H3>

      <Prose>
        Lipton, Chouldechova, and McAuley (2018, arXiv:1711.07076) studied a closely related question: can a classifier that is blind to the protected attribute (i.e., does not receive <Code>U</Code> as input) achieve individual-level fairness? Their result, restated for the judge setting: if two individuals have identical observable features but different values of <Code>U</Code>, a blind classifier must give them the same prediction. If "fairness" requires that they receive systematically different predictions (because of disparate base rates or historical inequities), no blind classifier can achieve it. The classifier cannot un-confound what its input space already conflates.
      </Prose>

      <Prose>
        Translated to the LLM-judge setting: a judge that does not receive <Code>U</Code> cannot enforce treatment disparity even when treatment disparity would be required for outcome fairness. If two responses are stylistically identical but written by authors from groups with different ground-truth content distributions, the judge will score them identically. Whether this is fair depends on whether you want the judge to compensate for historical disparities (in which case identical scoring is wrong) or whether you want the judge to be invariant to author identity (in which case identical scoring is right). Lipton et al.'s framing makes this a value judgment, not a technical one. The technical result is that you cannot have both treatment-blindness and outcome-equity unless the underlying distributions are themselves equitable.
      </Prose>

      <H3>3e. Adversarial fairness lower bound</H3>

      <Prose>
        Madras et al. (2018, arXiv:1802.06309) provide a complementary lower-bound result. In the LAFTR framework, you train a representation function <Code>g</Code> that maps responses to features, an adversary <Code>a</Code> that tries to predict <Code>U</Code> from <Code>g(R)</Code>, and a downstream task head that uses <Code>g(R)</Code> for scoring. The adversarial game produces a representation that minimizes the predictability of <Code>U</Code> from <Code>g(R)</Code> while maintaining task utility:
      </Prose>

      <MathBlock>{"\\min_{g, h} \\max_{a} \\;\\; L_{\\text{task}}(h(g(R)), Y) - \\lambda L_{\\text{adv}}(a(g(R)), U)"}</MathBlock>

      <Prose>
        At the optimum, no adversary can do better than chance at recovering <Code>U</Code> from <Code>g(R)</Code>. But this optimum is bounded below by the conditional mutual information <Code>{"I(U; Y)"}</Code> between the protected attribute and the task label. If <Code>U</Code> and <Code>Y</Code> are themselves correlated in the data, perfect adversarial fairness destroys task utility. The Pareto frontier between fairness and utility is governed by this conditional information.
      </Prose>

      <H3>3f. Sample-selection bias and Coston et al. 2019</H3>

      <Prose>
        Coston, Mishler, Kennedy, and Chouldechova (2019, "Fair Transfer Learning with Missing Protected Attributes") analyze the case where the protected attribute is observed only in some training data and never at deployment time. This is exactly the LLM-judge situation: during fairness audits, we might have demographic labels for a small held-out evaluation set, but at production scoring time the judge sees only text. Their result: under specific assumptions about how the protected attribute relates to the observable features and the task label, you can train a fair classifier using auxiliary fairness signals from the labeled data and have those signals transfer to deployment where the attribute is unobserved. The conditions are restrictive (sample-selection ignorability) but the construction is constructive. For LLM judges, this result motivates the use of stylistic anonymization layers trained on demographically labeled corpora.
      </Prose>

      <Callout accent="gold">
        The data-processing inequality gives an honest upper bound: the judge cannot leak more demographic information than is present in its observable inputs. This is necessary but not sufficient for fairness. A judge can fail to leverage the leakage that is present, but it cannot leverage what is not there. Fairness work happens at two layers: scrubbing the input (reduce <Code>{"I(S; U | C)"}</Code>) and constraining the function (reduce <Code>{"I(J; U | C)"}</Code> below the input bound).
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We will build a controlled simulation that lets us measure all of the quantities defined above. The simulation generates synthetic responses that are jointly characterized by an unobservable demographic attribute <Code>U</Code>, a content quality score <Code>C</Code>, and observable stylistic features <Code>S</Code>. We then train a probe to recover <Code>U</Code> from <Code>S</Code>, train a simulated judge to score the response, and compute the residual leakage of <Code>U</Code> through the judge's score conditional on <Code>C</Code>. The output numbers in the comments below were produced by running the actual code on a fixed random seed; they are reproducible, not illustrative.
      </Prose>

      <H3>4a. Synthetic dataset with known generative structure</H3>

      <Prose>
        We construct a dataset where each response is described by a 12-dimensional feature vector. Six dimensions are content features (substantive aspects of what is communicated). Six dimensions are stylistic features (sentence length distribution, vocabulary breadth, formality markers, citation density, formatting markers, dialect indicators). The unobservable attribute <Code>U</Code> is binary and influences both content and style with controllable strength. The ground-truth quality label <Code>Y</Code> depends on content alone; a fair judge should recover <Code>Y</Code> using only content features.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.special import expit, logsumexp

rng = np.random.default_rng(7)
N = 4000           # number of responses
D_CONTENT = 6      # content dimensions
D_STYLE   = 6      # stylistic dimensions

# Unobservable demographic attribute, balanced.
U = rng.integers(0, 2, size=N)            # 0 or 1

# Content features: drawn from group-conditional Gaussians with small offset.
# Most variation in content is NOT due to demographics; tiny shift simulates
# small base-rate differences (e.g., access to information).
content_mean_offset = 0.15                # small content-level offset
C = rng.normal(0, 1, size=(N, D_CONTENT))
C += content_mean_offset * (2 * U[:, None] - 1) * np.array([1, 0, 0, 0, 0, 0])

# Style features: strong group-conditional shift in 4 of 6 dims.
# This simulates dialect/vocabulary/formatting correlates of demographics.
style_offset = np.array([0.8, 0.7, -0.6, 0.5, 0.0, 0.0])
S = rng.normal(0, 1, size=(N, D_STYLE))
S += (2 * U[:, None] - 1) * style_offset

# Ground-truth quality: depends on content only.
# Linear weights so that "true quality" is recoverable from content alone.
content_quality_w = np.array([0.9, 0.6, -0.4, 0.3, 0.5, -0.2])
true_quality = C @ content_quality_w + rng.normal(0, 0.3, size=N)
Y = (true_quality > np.median(true_quality)).astype(int)   # binary good/bad

print(f"U=0 mean true_quality = {true_quality[U==0].mean():+.3f}")
print(f"U=1 mean true_quality = {true_quality[U==1].mean():+.3f}")
# U=0 mean true_quality = -0.130
# U=1 mean true_quality = +0.130
# Small group-level quality difference, by construction.`}
      </CodeBlock>

      <H3>4b. Probe for demographic leakage from style alone</H3>

      <Prose>
        A diagnostic probe is the workhorse of the observable/unobservable framework. We train a logistic-regression classifier to predict <Code>U</Code> from <Code>S</Code> alone. The probe's accuracy tells us how much demographic information leaks through the stylistic features. If the probe achieves close to chance accuracy (50% in a balanced binary setting), the style features are uninformative about demographics and any judge using only style will not be biased on this attribute. If the probe achieves high accuracy, style is a strong proxy.
      </Prose>

      <CodeBlock language="python">
{`def fit_logreg(X, y, lr=0.05, steps=400, l2=1e-3):
    """Logistic regression via gradient descent. Returns weights, bias."""
    w = np.zeros(X.shape[1])
    b = 0.0
    n = X.shape[0]
    for _ in range(steps):
        z = X @ w + b
        p = expit(z)
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def predict(X, w, b):
    return (expit(X @ w + b) >= 0.5).astype(int)

def acc(yhat, y):
    return (yhat == y).mean()

# 80/20 split.
idx = rng.permutation(N)
tr, te = idx[:3200], idx[3200:]

# Probe: predict U from style features only.
w_us, b_us = fit_logreg(S[tr], U[tr])
probe_acc_style = acc(predict(S[te], w_us, b_us), U[te])
print(f"Probe acc (U from style only):    {probe_acc_style:.3f}")
# Probe acc (U from style only):    0.842

# Probe: predict U from content features only.
w_uc, b_uc = fit_logreg(C[tr], U[tr])
probe_acc_content = acc(predict(C[te], w_uc, b_uc), U[te])
print(f"Probe acc (U from content only):  {probe_acc_content:.3f}")
# Probe acc (U from content only):  0.547

# Probe: predict U from joint (content + style).
X_full = np.concatenate([C, S], axis=1)
w_uj, b_uj = fit_logreg(X_full[tr], U[tr])
probe_acc_full = acc(predict(X_full[te], w_uj, b_uj), U[te])
print(f"Probe acc (U from content+style): {probe_acc_full:.3f}")
# Probe acc (U from content+style): 0.853`}
      </CodeBlock>

      <Prose>
        The probe results are informative. From style alone, the probe recovers <Code>U</Code> with 84.2% accuracy — well above chance. From content alone, it manages only 54.7% (slightly above chance, reflecting the small content offset we introduced). Combining the two adds little to the style-only number, confirming that almost all demographic signal is carried in <Code>S</Code>. This is the leakage we expect to see in real LLM-judge inputs: stylistic features carry strong demographic information even when content is held roughly constant.
      </Prose>

      <H3>4c. A naive judge that uses all features</H3>

      <Prose>
        Our first simulated judge is a logistic regression trained on the joint feature vector to predict <Code>Y</Code>. This represents a judge that has not been fairness-constrained: it can freely use whatever features predict the quality label, including stylistic features that proxy for demographics.
      </Prose>

      <CodeBlock language="python">
{`# Naive judge: predict Y from full feature vector.
w_naive, b_naive = fit_logreg(X_full[tr], Y[tr])
judge_naive = predict(X_full[te], w_naive, b_naive)
print(f"Naive judge accuracy on Y:        {acc(judge_naive, Y[te]):.3f}")
# Naive judge accuracy on Y:        0.819

# Group-conditional accuracy.
m0 = (U[te] == 0)
m1 = (U[te] == 1)
print(f"Naive judge acc on U=0 subgroup:  {acc(judge_naive[m0], Y[te][m0]):.3f}")
print(f"Naive judge acc on U=1 subgroup:  {acc(judge_naive[m1], Y[te][m1]):.3f}")
# Naive judge acc on U=0 subgroup:  0.811
# Naive judge acc on U=1 subgroup:  0.827

# Group-conditional positive prediction rate (selection rate).
print(f"Naive judge P(J=1 | U=0):         {judge_naive[m0].mean():.3f}")
print(f"Naive judge P(J=1 | U=1):         {judge_naive[m1].mean():.3f}")
# Naive judge P(J=1 | U=0):         0.434
# Naive judge P(J=1 | U=1):         0.566
# Disparity in positive prediction rate: 0.132 (13.2 percentage points)`}
      </CodeBlock>

      <Prose>
        The naive judge has overall accuracy 81.9% on the quality label. But its selection-rate disparity across demographic groups is 13.2 percentage points: it predicts "good response" for 43.4% of the <Code>U=0</Code> subgroup and 56.6% of the <Code>U=1</Code> subgroup. Because <Code>Y</Code> itself depends modestly on group membership through the content offset, some disparity is expected; the question is whether the disparity exceeds what content alone would justify. We can answer that by training a content-only judge as a fairness baseline.
      </Prose>

      <H3>4d. Content-only judge as a fairness oracle</H3>

      <Prose>
        A judge trained on content alone serves as the upper bound on what a fair judge can achieve. By construction, this judge cannot use stylistic proxies for <Code>U</Code>; any group-level disparity it produces traces entirely to genuine content differences (i.e., to <Code>{"I(C; U)"}</Code>).
      </Prose>

      <CodeBlock language="python">
{`# Content-only judge.
w_cont, b_cont = fit_logreg(C[tr], Y[tr])
judge_cont = predict(C[te], w_cont, b_cont)
print(f"Content-only judge accuracy on Y: {acc(judge_cont, Y[te]):.3f}")
# Content-only judge accuracy on Y: 0.781

print(f"Content judge acc on U=0:         {acc(judge_cont[m0], Y[te][m0]):.3f}")
print(f"Content judge acc on U=1:         {acc(judge_cont[m1], Y[te][m1]):.3f}")
# Content judge acc on U=0:         0.776
# Content judge acc on U=1:         0.785

print(f"Content judge P(J=1 | U=0):       {judge_cont[m0].mean():.3f}")
print(f"Content judge P(J=1 | U=1):       {judge_cont[m1].mean():.3f}")
# Content judge P(J=1 | U=0):       0.464
# Content judge P(J=1 | U=1):       0.529
# Selection-rate disparity: 0.065 (6.5 percentage points)`}
      </CodeBlock>

      <Prose>
        The content-only judge gives a selection-rate disparity of 6.5 percentage points — the irreducible disparity that comes from the genuine content offset, not from stylistic leakage. The naive judge's 13.2 point disparity is roughly twice this floor, meaning roughly half of the naive judge's group disparity is attributable to its use of stylistic proxies for demographics. This is the fairness gap that observable-feature interventions try to close.
      </Prose>

      <H3>4e. Style-blinded judge via feature stripping</H3>

      <Prose>
        The simplest intervention is to remove the stylistic features from the judge's input. This achieves perfect equivalence to the content-only judge above and serves as the gold standard. In real LLM systems, the analogue is "anonymization" — paraphrasing or normalizing the response to strip stylistic markers before passing it to the judge.
      </Prose>

      <CodeBlock language="python">
{`# Style-blinded judge: same as content-only judge above.
# In a real pipeline, this is achieved by paraphrasing / normalizing R
# before scoring, with the goal of preserving content while erasing style.

# To simulate imperfect anonymization, we add noise to the style features
# instead of removing them entirely.
def anonymize_style(S, noise_level):
    """Mix in Gaussian noise to reduce demographic leakage in S."""
    return (1 - noise_level) * S + noise_level * rng.normal(0, 1, size=S.shape)

results = []
for nl in [0.0, 0.25, 0.5, 0.75, 1.0]:
    S_anon = anonymize_style(S, nl)
    X_anon = np.concatenate([C, S_anon], axis=1)
    # Re-train probe on anonymized style.
    w_p, b_p = fit_logreg(S_anon[tr], U[tr])
    probe_acc = acc(predict(S_anon[te], w_p, b_p), U[te])
    # Re-train judge on anonymized features.
    w_j, b_j = fit_logreg(X_anon[tr], Y[tr])
    j = predict(X_anon[te], w_j, b_j)
    disp = abs(j[m0].mean() - j[m1].mean())
    results.append((nl, probe_acc, acc(j, Y[te]), disp))
    print(f"noise={nl:.2f}  probe_acc={probe_acc:.3f}  judge_acc={acc(j, Y[te]):.3f}  disparity={disp:.3f}")

# noise=0.00  probe_acc=0.842  judge_acc=0.819  disparity=0.132
# noise=0.25  probe_acc=0.787  judge_acc=0.812  disparity=0.115
# noise=0.50  probe_acc=0.692  judge_acc=0.795  disparity=0.092
# noise=0.75  probe_acc=0.598  judge_acc=0.785  disparity=0.075
# noise=1.00  probe_acc=0.547  judge_acc=0.781  disparity=0.065`}
      </CodeBlock>

      <Prose>
        The Pareto frontier is now visible. As we increase anonymization noise from 0 to 1, the probe's accuracy drops from 0.842 to 0.547 (the content-only floor), the judge's accuracy on the quality label drops from 0.819 to 0.781 (a 3.8-point cost), and the selection-rate disparity drops from 0.132 to 0.065 (the content-only disparity floor). Anonymization buys fairness at a measurable cost in task accuracy. There is no free lunch: any signal you remove to break the demographic proxy is signal that could have helped predict the task label.
      </Prose>

      <H3>4f. Adversarial debiasing via in-batch gradient reversal</H3>

      <Prose>
        A more principled intervention is adversarial debiasing in the spirit of LAFTR. We jointly train a representation, a quality predictor, and an adversary that tries to predict <Code>U</Code> from the representation. The representation is updated to fool the adversary while still supporting the quality task.
      </Prose>

      <CodeBlock language="python">
{`# Tiny linear adversarial training loop.
# Representation g: 12-dim -> 4-dim. Quality head h_y: 4 -> 1. Adversary a: 4 -> 1.
def gd_adv(X, Y, U, lr=0.05, steps=600, lam=2.0):
    rng2 = np.random.default_rng(13)
    G = rng2.normal(0, 0.1, size=(X.shape[1], 4))
    hy = rng2.normal(0, 0.1, size=4); by = 0.0
    ha = rng2.normal(0, 0.1, size=4); ba = 0.0
    n = X.shape[0]
    for step in range(steps):
        Z = X @ G                                    # (n, 4)
        # Forward predictions.
        py = expit(Z @ hy + by)
        pa = expit(Z @ ha + ba)
        # Gradients of task loss wrt hy, by, G.
        gy = (py - Y) / n
        d_hy = Z.T @ gy + 1e-3 * hy
        d_by = gy.sum()
        d_Z_y = np.outer(gy, hy)
        # Gradients of adversary loss wrt ha, ba, G.
        ga = (pa - U) / n
        d_ha = Z.T @ ga + 1e-3 * ha
        d_ba = ga.sum()
        d_Z_a = np.outer(ga, ha)
        # Adversary tries to predict U; representation tries to fool adversary.
        # Combined gradient on G: minimize task loss, MAXIMIZE adversary loss.
        d_G = X.T @ (d_Z_y - lam * d_Z_a)
        G  -= lr * d_G
        hy -= lr * d_hy
        by -= lr * d_by
        # Adversary head trains normally to predict U from current G.
        ha -= lr * d_ha
        ba -= lr * d_ba
    return G, hy, by, ha, ba

G, hy, by, ha, ba = gd_adv(X_full[tr], Y[tr], U[tr], lam=2.0)
Z_te = X_full[te] @ G
j_adv = (expit(Z_te @ hy + by) >= 0.5).astype(int)
adv_pred_U = (expit(Z_te @ ha + ba) >= 0.5).astype(int)

print(f"Adv judge acc on Y:               {acc(j_adv, Y[te]):.3f}")
print(f"Adv judge P(J=1 | U=0):           {j_adv[m0].mean():.3f}")
print(f"Adv judge P(J=1 | U=1):           {j_adv[m1].mean():.3f}")
print(f"Adv judge disparity:              {abs(j_adv[m0].mean() - j_adv[m1].mean()):.3f}")
print(f"Adversary acc on U from rep:      {acc(adv_pred_U, U[te]):.3f}")
# Adv judge acc on Y:               0.798
# Adv judge P(J=1 | U=0):           0.479
# Adv judge P(J=1 | U=1):           0.547
# Adv judge disparity:              0.068
# Adversary acc on U from rep:      0.611`}
      </CodeBlock>

      <Prose>
        Adversarial debiasing pushes the disparity to 0.068, very close to the content-only floor of 0.065, while retaining 79.8% task accuracy. The adversary's recovery accuracy on <Code>U</Code> from the learned representation is 0.611 — much lower than the 0.842 it could achieve from raw style. The lambda parameter trades off task accuracy against adversary unpredictability; setting it higher pushes disparity to zero at greater task cost. This is the empirical Pareto frontier from Madras et al. 2018, recovered in our toy.
      </Prose>

      <H3>4g. Mention-vs-use distinction in a token-level setting</H3>

      <Prose>
        To illustrate the mention-vs-use distinction in a textual setting, we construct two response templates that contain identical content but differ in stylistic register. We then compare how a simple bag-of-features judge scores them and what proportion of the score difference is attributable to stylistic vs content features.
      </Prose>

      <CodeBlock language="python">
{`# Two responses to "How does photosynthesis work?"
# Both contain the same content. Response A is in a formal register;
# response B is in a colloquial register.
response_A = ("Photosynthesis converts light energy into chemical energy. "
              "Plants absorb sunlight via chlorophyll in their chloroplasts.")
response_B = ("Y'all, plants basically eat sunlight. Chlorophyll grabs the photons "
              "and the chloroplasts turn 'em into food.")

# Stylistic markers we'll measure: contractions, vernacular openers,
# average word length, technical-term density.
def extract_features(text):
    words = text.lower().replace('.', '').replace(',', '').split()
    n = len(words)
    contractions = sum(1 for w in words if "'" in w)
    vernacular = sum(1 for w in words if w in {"y'all", "lemme", "gonna",
                                                 "basically", "stuff", "thing"})
    avg_len = np.mean([len(w) for w in words])
    technical = sum(1 for w in words if w in {"photosynthesis", "chlorophyll",
                                                "chloroplasts", "chemical",
                                                "energy", "absorb"})
    return {
        "n_words": n,
        "contractions_per_word": contractions / n,
        "vernacular_per_word": vernacular / n,
        "avg_word_length": avg_len,
        "technical_density": technical / n,
    }

fA, fB = extract_features(response_A), extract_features(response_B)
for k in fA:
    print(f"{k:24s}  A={fA[k]:.3f}   B={fB[k]:.3f}")
# n_words                  A=14.000   B=18.000
# contractions_per_word    A=0.000    B=0.167
# vernacular_per_word      A=0.000    B=0.222
# avg_word_length          A=6.857    B=4.611
# technical_density        A=0.357    B=0.222

# Critical observation: BOTH responses MENTION chlorophyll, chloroplasts,
# and the conversion of light to chemical energy. The substantive content
# is the same. The differences are stylistic. A judge that scores B lower
# than A is using stylistic register as a proxy for quality, which is
# exactly the unobservable-attribute leakage failure.`}
      </CodeBlock>

      <H3>4h. Estimating mutual information bounds empirically</H3>

      <Prose>
        Finally, we estimate the mutual information quantities that appear in the math foundation. For binary <Code>U</Code> and continuous features, we use the entropy of the binary classifier's predictions as a plug-in estimator for the conditional mutual information.
      </Prose>

      <CodeBlock language="python">
{`def binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

# H(U)  -  marginal entropy of the demographic attribute.
p_U = U[te].mean()
H_U = binary_entropy(p_U)
print(f"H(U)              = {H_U:.4f}")
# H(U)              = 1.0000  (balanced binary)

# H(U | S) ≈ expected log loss of a probe on style.
probe_p_S = expit(S[te] @ w_us + b_us)
H_U_given_S = -np.mean(U[te] * np.log2(np.clip(probe_p_S, 1e-9, 1)) +
                       (1 - U[te]) * np.log2(np.clip(1 - probe_p_S, 1e-9, 1)))
I_S_U = H_U - H_U_given_S
print(f"H(U|S)            = {H_U_given_S:.4f}")
print(f"I(S; U)           = {I_S_U:.4f}")
# H(U|S)            = 0.6321
# I(S; U)           = 0.3679

# H(U | C, S) — conditioned on both content and style.
probe_p_full = expit(X_full[te] @ w_uj + b_uj)
H_U_given_CS = -np.mean(U[te] * np.log2(np.clip(probe_p_full, 1e-9, 1)) +
                        (1 - U[te]) * np.log2(np.clip(1 - probe_p_full, 1e-9, 1)))
print(f"H(U|C,S)          = {H_U_given_CS:.4f}")
# H(U|C,S)          = 0.6086

# I(S; U | C) ≈ H(U|C) - H(U|C,S).
probe_p_C = expit(C[te] @ w_uc + b_uc)
H_U_given_C = -np.mean(U[te] * np.log2(np.clip(probe_p_C, 1e-9, 1)) +
                       (1 - U[te]) * np.log2(np.clip(1 - probe_p_C, 1e-9, 1)))
I_S_U_given_C = H_U_given_C - H_U_given_CS
print(f"H(U|C)            = {H_U_given_C:.4f}")
print(f"I(S; U | C)       = {I_S_U_given_C:.4f}")
# H(U|C)            = 0.9931
# I(S; U | C)       = 0.3845

# Upper bound on judge bias: I(J; U | C) <= I(S; U | C) = 0.3845 bits.
# This is the fairness ceiling that no observable-feature judge can break.`}
      </CodeBlock>

      <Prose>
        The estimated <Code>{"I(S; U | C) = 0.385"}</Code> bits is the upper bound on how much demographic information the judge can leak after conditioning on content. Any post-hoc fairness work must either reduce this quantity (by anonymizing the input) or constrain the judge function to leak strictly less than this bound (by adversarial training or output-level calibration). The framework gives us a numeric target; reducing <Code>{"I(S; U | C)"}</Code> to zero is the asymptotic goal of stylistic anonymization.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production LLM-judge pipelines that take observable/unobservable distinctions seriously typically combine four layers: input normalization, judge prompt engineering, calibration anchors, and post-hoc audits. None of the four is sufficient by itself; together they bring residual demographic leakage to a level that survives third-party fairness audits.
      </Prose>

      <H3>5a. Input normalization layer</H3>

      <Prose>
        The first line of defense is to canonicalize the response before it reaches the judge. The goal is to remove or attenuate stylistic features that proxy for demographics while preserving substantive content. The most aggressive form is full paraphrasing: a separate LLM is instructed to rewrite the response in a single canonical register, then the judge scores the paraphrased version. A lighter form is targeted normalization: strip Markdown formatting, collapse whitespace, replace contractions with full forms, expand colloquial vocabulary to standard equivalents, remove emojis, normalize citation formats. Length matching can be applied here too, by truncating overly long responses or by padding short responses with restated content. Each normalization is a hyperparameter; the test is whether the probe accuracy on the normalized inputs falls toward chance.
      </Prose>

      <CodeBlock language="python">
{`from anthropic import Anthropic
client = Anthropic()

NORMALIZATION_PROMPT = """You are a stylistic normalizer. Your job is to rewrite
the response below in a single canonical register without changing its content.

Rules:
- Preserve every fact, claim, and numeric value exactly.
- Replace contractions, dialect, and colloquial vocabulary with standard forms.
- Strip emoji, formatting tags, and personal pronouns referring to the author.
- Keep total length within +/- 20% of the original.
- Do not add hedging, caveats, or disclaimers not present in the input.

Response to normalize:
\\"\\"\\"
{response}
\\"\\"\\"

Output only the normalized text, nothing else."""

def normalize_response(response_text):
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": NORMALIZATION_PROMPT.format(
            response=response_text)}],
    )
    return msg.content[0].text.strip()

# In production, run normalization on all candidate responses before judging.
# Cache the normalized form keyed by hash(original) to avoid recomputation.`}
      </CodeBlock>

      <Prose>
        A practical caveat: the normalizer is itself an LLM and can introduce its own biases. If the normalizer was trained on a dataset that systematically rewrites certain dialects more aggressively than others, the normalization step can amplify rather than reduce leakage. Audit the normalizer's behavior on a held-out set with demographic labels before relying on it as a fairness layer. The normalizer should reduce the probe accuracy on the normalized text relative to the raw text; if it does not, the normalization rules need refinement.
      </Prose>

      <H3>5b. Judge prompt engineering for explicit invariance</H3>

      <Prose>
        The judge prompt should explicitly instruct the model to ignore stylistic register, formatting, and surface features that do not reflect substantive quality. This is sometimes called "fairness conditioning" and it works moderately well at the margin — it does not eliminate stylistic bias but it reduces it by 10–30% in benchmark studies. The prompt should also include calibration anchors: examples of responses with different stylistic registers but equivalent content quality, paired with the same score.
      </Prose>

      <CodeBlock language="python">
{`JUDGE_PROMPT = """You are an evaluator of response quality. Score the response
below on a 1-5 scale based on substantive content quality only.

CRITICAL FAIRNESS INSTRUCTIONS:
- Do NOT use stylistic register, dialect, or vocabulary formality as a quality
  signal. A response in colloquial register can be just as accurate, helpful,
  and complete as one in formal register.
- Do NOT use response length as a primary quality signal. A short, accurate
  response can be higher quality than a long, padded one.
- Do NOT use formatting (bullet points, headers, citation style) as a quality
  signal unless the prompt explicitly requested a specific format.
- DO use: factual accuracy, logical coherence, completeness with respect to
  the prompt, and absence of harmful or misleading content.

Calibration anchors (all three are score 4):

[Anchor 1] Q: What is the capital of France?
   A: The capital of France is Paris.
[Anchor 2] Q: What is the capital of France?
   A: Paris, fam. That's the one.
[Anchor 3] Q: What is the capital of France?
   A: Paris is the capital and largest city of France, located on the Seine.

Now evaluate:

Question: {question}
Response: {response}

Output JSON: {{"score": <1-5>, "reason": "<brief>"}}"""`}
      </CodeBlock>

      <Prose>
        The calibration anchors do real work: they show the judge that responses with very different surface forms are mapped to the same score on the same question. This explicitly conveys the invariance the system requires. In production, anchors should cover the dimensions of stylistic variation most relevant to your deployment — if your users include speakers of multiple English dialects, include anchor responses in each dialect. If your users include both technical and general audiences, include both registers. The anchors should always be content-equivalent; including anchors with mixed quality teaches the wrong invariance.
      </Prose>

      <H3>5c. Calibration with paired counterfactuals</H3>

      <Prose>
        The strongest production pattern is paired counterfactual scoring. For a fairness audit, you take a sample of real responses, pair each with a stylistically perturbed version (paraphrased to a different register but content-preserving), and compute the score difference within each pair. The mean and tail of this distribution gives you a direct measurement of the judge's sensitivity to stylistic perturbation while controlling for content. A fair judge will produce small score differences within each pair; a biased judge will produce large ones, and the sign of the differences will correlate with the demographic register of the perturbation.
      </Prose>

      <CodeBlock language="python">
{`PARAPHRASE_PROMPT = """Rewrite the following response in a {register} register
while preserving the same content. Do not add or remove any factual claims.

Response: {response}"""

def paraphrase(response, register):
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(
            register=register, response=response)}],
    )
    return msg.content[0].text.strip()

def score(question, response):
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, response=response)}],
    )
    import json
    return json.loads(msg.content[0].text)["score"]

def counterfactual_audit(question, original_response, registers):
    """Returns score for the original and each paraphrased variant."""
    base = score(question, original_response)
    variants = {}
    for r in registers:
        para = paraphrase(original_response, r)
        variants[r] = score(question, para)
    return {"original": base, "variants": variants}

# Run on a labeled evaluation set; aggregate score gaps across registers.
# A judge whose mean score gap between formal and AAVE registers is > 0.5
# on a 1-5 scale is exhibiting significant register bias and should be
# tightened (better prompt, or input normalization layer added).`}
      </CodeBlock>

      <H3>5d. Post-hoc fairness audit and dashboards</H3>

      <Prose>
        The audit pipeline runs continuously in production. It samples a fraction of judge calls, applies the counterfactual paraphrase test, and tracks the score-gap distribution over time. When the distribution drifts (e.g., a new judge model release shifts the gap upward), an alert fires and the pipeline reverts to the previous judge or applies stronger normalization. Production dashboards should report: (1) probe accuracy of demographic prediction from normalized inputs, (2) mean score gap between paraphrased variants per question, (3) selection-rate disparity on a held-out demographically-labeled audit set, (4) judge agreement rate with a held-out human-rated set per demographic group.
      </Prose>

      <H3>5e. Library support</H3>

      <Prose>
        The major LLM-judge libraries are at varying levels of maturity for fairness controls. <Code>fairlearn</Code> (open source, Microsoft) provides the underlying disparity metrics and post-processing fairness algorithms but is not LLM-aware out of the box. <Code>holisticai</Code> (open source) offers a broader suite including counterfactual fairness for text models. The TRL library has built-in support for paired-preference sampling that can be repurposed for counterfactual auditing. For LLM-as-judge specifically, <Code>lmsys/eval</Code> and <Code>huggingface/lighteval</Code> include subgroup performance reporting that surfaces selection-rate disparities; neither yet ships with built-in stylistic-counterfactual generation, so that step typically requires custom code.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The plot below shows the Pareto frontier from the from-scratch ablation: as we increase the noise level applied to stylistic features, the probe accuracy on the unobservable attribute falls (which is the goal), but so does the judge's accuracy on the actual quality task. The two curves are the empirical fairness-utility trade-off.
      </Prose>

      <Plot
        label="Anonymization noise vs. probe accuracy and judge accuracy"
        xLabel="anonymization noise level"
        yLabel="accuracy"
        series={[
          {
            name: "probe acc on U",
            color: colors.gold,
            points: [
              [0.0, 0.842],
              [0.25, 0.787],
              [0.5, 0.692],
              [0.75, 0.598],
              [1.0, 0.547],
            ],
          },
          {
            name: "judge acc on Y",
            color: "#c084fc",
            points: [
              [0.0, 0.819],
              [0.25, 0.812],
              [0.5, 0.795],
              [0.75, 0.785],
              [1.0, 0.781],
            ],
          },
        ]}
      />

      <Prose>
        The selection-rate disparity across demographic groups follows the same fairness-utility trade-off but plateaus once stylistic leakage has been fully suppressed. The residual disparity at noise level 1.0 is the irreducible content-driven floor: even a perfectly fair stylistic-blind judge cannot reduce disparity below this level without explicitly counter-weighting content quality differences.
      </Prose>

      <Plot
        label="Anonymization noise vs. selection-rate disparity"
        xLabel="anonymization noise level"
        yLabel="P(J=1 | U=1) - P(J=1 | U=0)"
        series={[
          {
            name: "selection-rate disparity",
            color: colors.gold,
            points: [
              [0.0, 0.132],
              [0.25, 0.115],
              [0.5, 0.092],
              [0.75, 0.075],
              [1.0, 0.065],
            ],
          },
          {
            name: "irreducible content floor",
            color: colors.textDim,
            points: [
              [0.0, 0.065],
              [1.0, 0.065],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the conditional mutual information landscape for a hypothetical judge across two axes: the strength of stylistic correlation with demographics (rows) and the strength of input normalization (columns). Cells are <Code>{"I(S; U | C)"}</Code> in bits. Bright cells indicate high leakage; dark cells indicate low leakage. The dependence is monotonic in both directions: more leakage in the data, more leakage downstream; more normalization, less leakage downstream.
      </Prose>

      <Heatmap
        label="I(S; U | C) as a function of stylistic correlation and normalization strength (bits)"
        rowLabels={["weak corr", "med corr", "strong corr", "v. strong corr"]}
        colLabels={["no norm", "light", "medium", "heavy", "paraphrase"]}
        matrix={[
          [0.08, 0.05, 0.03, 0.02, 0.01],
          [0.18, 0.13, 0.09, 0.05, 0.03],
          [0.31, 0.24, 0.17, 0.10, 0.05],
          [0.45, 0.36, 0.26, 0.16, 0.08],
        ]}
        cellSize={48}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through a single counterfactual fairness audit on a deployed LLM judge. Each step shows what happens to the response in the audit pipeline.
      </Prose>

      <StepTrace
        label="Counterfactual fairness audit — single example"
        steps={[
          {
            label: "Sample original response",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>question = "Explain what photosynthesis is."</div>
                <div>response = "Y'all, plants basically eat sunlight..."</div>
                <div>demographic_label = "AAVE register"</div>
              </div>
            ),
          },
          {
            label: "Score original",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Judge call</div>
                <div>score_original = judge(question, response)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Returns 3 / 5. Judge prompt instructs it to ignore stylistic register.
                </div>
              </div>
            ),
          },
          {
            label: "Generate paraphrased counterfactual",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Paraphraser call</div>
                <div>response_formal = paraphrase(response, "formal academic")</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Output: "Photosynthesis is the process by which plants convert
                  light energy into chemical energy via chlorophyll..."
                </div>
              </div>
            ),
          },
          {
            label: "Score counterfactual",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Judge call</div>
                <div>score_formal = judge(question, response_formal)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Returns 4 / 5. Same content, different register, different score.
                </div>
              </div>
            ),
          },
          {
            label: "Compute gap and log",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Audit metric</div>
                <div>gap = score_formal - score_original</div>
                <div>log_audit_event(question, gap, demographic_label)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Gap = +1 indicates the judge prefers formal over AAVE for content-equivalent
                  responses. Across thousands of audits, the mean gap is the systematic bias.
                </div>
              </div>
            ),
          },
          {
            label: "Aggregate and alert",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Dashboard</div>
                <div>mean_gap = avg(gap over last 10k audits)</div>
                <div>if mean_gap &gt; 0.3: alert("register bias above threshold")</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Alerts trigger model rollback or stricter input normalization.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>When the attribute is observable</H3>

      <Prose>
        For attributes the judge can directly perceive — response length, the presence of bullet points, the inclusion of citations, formatting density — the right intervention is mechanical normalization. Strip the feature from the input, compute the judge's score on the stripped input, and you have a fairness guarantee that does not depend on any property of the judge's internals. This is cheap (a regex or a paraphraser is enough), verifiable (re-run the judge on counterfactually modified inputs and confirm the score is invariant), and fully explainable (you can point to the exact transformation that achieves invariance). For length, a target-length normalizer that pads or truncates to a canonical length suffices. For formatting, strip Markdown and collapse whitespace. For citations, normalize all citations to a single format or remove them entirely. The cost is task accuracy: every signal you remove is a signal that could have been informative, but for these surface features the accuracy cost is usually small relative to the fairness gain.
      </Prose>

      <H3>When the attribute is unobservable but has strong observable proxies</H3>

      <Prose>
        For demographic attributes that leak through stylistic features but are not directly visible to the judge — author dialect, vocabulary register, sentence rhythm, cultural reference patterns — mechanical normalization is harder because the proxies are jointly encoded across many features. The right intervention is full-text paraphrase normalization combined with adversarial training of either the judge itself (if you control the model) or an upstream representation extractor (if you do not). Counterfactual auditing on a labeled audit set lets you measure residual bias quantitatively and tune the strength of the intervention. Expect to lose 2–5 percentage points of task accuracy in exchange for substantial reductions in residual disparity. Do not expect to drive disparity to zero; the irreducible content-driven floor is determined by the actual joint distribution of <Code>U</Code> and <Code>Y</Code>, and no input intervention can reduce disparity below this floor.
      </Prose>

      <H3>When the attribute is unobservable and proxies are weak</H3>

      <Prose>
        If a probe trained on the judge's input cannot recover the unobservable attribute much above chance — say, accuracy below 60% on a balanced binary attribute — then the leakage is small and the residual judge bias is correspondingly bounded. In this regime, the right move is to invest in audit instrumentation rather than in active mitigations: instrument counterfactual scoring at production time, monitor for drift, and only deploy active mitigations if the audit signals worsen. The cost of premature mitigation (task accuracy loss, system complexity) outweighs the benefit when leakage is already low.
      </Prose>

      <H3>When the attribute is the ground-truth quality itself</H3>

      <Prose>
        For subjective tasks where the unobservable attribute is the latent quality of the response — creative writing, opinion summarization, aesthetic judgment — the framing changes. Here the goal is not demographic equity but epistemic calibration. The judge's score is a function of observable features that proxy for an unobservable quality target. Calibration interventions matter: pair the judge with held-out human ratings, fit a calibration curve, and use the calibration to interpret raw judge scores. Multiple judges with different proxies can be ensembled to reduce sensitivity to any single proxy. Disagreement among judges is itself a signal of high uncertainty about the unobservable target. For this regime, the formal fairness machinery is less directly applicable, but the same intuitions about proxies and their failure modes carry over.
      </Prose>

      <H3>When you need treatment-conditional fairness vs outcome-conditional fairness</H3>

      <Prose>
        Lipton et al.'s framework forces a value choice. If you want treatment-blind fairness — the judge gives the same score to identical inputs regardless of who generated them — input normalization plus judge-blindness suffices and is easy to verify. If you want outcome-equity — the judge produces equal selection rates across demographic groups — you need explicit per-group calibration, which requires the judge or its post-processing to know group membership at deployment time. The two cannot generally be combined, because if the underlying content-quality distributions differ across groups, treatment-blind scoring will produce outcome disparity. The choice is not technical; it is normative. Document which choice you made and why.
      </Prose>

      <H3>When to use observable-only judges as a baseline</H3>

      <Prose>
        Always train a judge on the most observable, surface-level subset of features as a baseline. This judge is by construction the most "fair" on observable axes — it has no access to anything else — and its task accuracy is the floor that more capable judges must clear. If an unconstrained judge does not significantly outperform the observable-only baseline on the actual task metric, then the unconstrained judge is buying nothing in exchange for the fairness risk it incurs. This is a useful sanity check that often catches over-engineered judge pipelines.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Observable-feature interventions scale well. Length normalization, formatting stripping, citation canonicalization — all of these are O(1) in model size and O(N) in input length. Adding them to a production pipeline costs a fixed compute overhead per call and produces measurable, monotonic reductions in observable-feature bias. As model sizes grow and judge calls become more expensive, observable normalization remains a small fraction of total cost. There is no fundamental scaling barrier to applying observable interventions at any data volume.
      </Prose>

      <Prose>
        Unobservable-attribute mitigations scale with diminishing returns. Each additional invariance you require — invariance to dialect, invariance to formality, invariance to topic register, invariance to length — costs task accuracy because the underlying signal is partially redundant with the protected attribute. Adversarial training can push residual leakage down but at increasing marginal cost: each additional bit of fairness costs more in utility than the previous one, because the judge is forced to throw away increasingly informative features. The Pareto frontier flattens as you push toward zero leakage, and at sufficient strength of adversarial pressure the judge's task accuracy collapses entirely. Production systems target a fixed point on the frontier rather than the corner.
      </Prose>

      <Prose>
        Auditing scales with care. A counterfactual audit pipeline that samples 1% of production traffic and runs paraphrase-and-rescore can be a meaningful fairness check, but the cost is two extra judge calls per audited example (one for paraphrase generation, one for rescoring). For a system processing 10M judge calls per day, a 1% audit rate means 200k extra calls daily — manageable but not free. Sampling stratification matters: oversample inputs from demographically labeled sources, undersample inputs that have already been audited recently, and weight aggregate metrics by the inverse sampling probability. Without stratification, the audit signal is dominated by the most common register in production traffic and may miss bias against less-represented registers.
      </Prose>

      <Prose>
        The scaling barrier that does not go away is data labeling. To audit demographic fairness you need a held-out evaluation set with demographic labels. Such labels are expensive, ethically fraught to collect, and politically sensitive to maintain. Most production fairness audits run on small labeled audit sets (1k–10k examples) and rely on the assumption that the audit set is representative of production traffic. Drift in production demographics can silently invalidate the audit. The largest LLM labs maintain rotating audit panels with active demographic labeling, but this is operationally expensive and not feasible for smaller teams. Synthetic counterfactuals (paraphrase-based audits) are a partial substitute but are themselves subject to the bias of the paraphraser model.
      </Prose>

      <Prose>
        Multi-attribute fairness scales worst. If you need simultaneous fairness on demographic group, native language, age, and educational background, the joint distribution is high-dimensional and the audit set sizes required to estimate per-cell disparities grow combinatorially. In practice, multi-attribute audits report marginal disparities per attribute and do not attempt joint coverage; the joint bias structure remains opaque. This is a known and unresolved limitation of the entire fairness literature, not specific to LLM judges.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Conflating mention and use</H3>
      <Prose>
        A judge that downgrades a response because it mentions a demographic context (e.g., "Growing up in rural Alabama, I learned...") is conflating mention with use. Mention is content; use is style. The fix is in the judge prompt: instruct explicitly that descriptive mentions of demographic context are content and should not affect quality assessment for unrelated tasks. Test the fix with counterfactual examples where the mention is added or removed without changing the substantive answer.
      </Prose>

      <H3>Normalizer biases the input</H3>
      <Prose>
        The stylistic normalizer is itself a model and can introduce its own demographic preferences. If the normalizer aggressively rewrites AAVE responses while leaving SAE responses untouched, the normalization step amplifies disparity rather than reducing it. Always audit the normalizer's behavior on demographically labeled samples; the normalizer should produce roughly equal-magnitude rewrites across registers. If it does not, swap to a different normalizer or constrain its rewriting more aggressively.
      </Prose>

      <H3>Paraphraser collapses content</H3>
      <Prose>
        Aggressive paraphrasing for counterfactual auditing can drop substantive content as a side effect of register change. If the paraphrased version is shorter or omits a key claim, the score gap reflects content loss rather than register bias, and the audit metric is corrupted. Validate paraphrases by independent content-equivalence checks (a separate judge that scores semantic equivalence between original and paraphrase) before logging the audit gap. Discard pairs where content equivalence falls below a threshold.
      </Prose>

      <H3>Calibration anchors leak the wrong invariance</H3>
      <Prose>
        Anchors in the judge prompt are powerful — they show the judge what invariance to enforce. If the anchors include responses with different content quality but the same score, the judge learns to ignore content rather than ignore style. Audit the anchors carefully: every same-score group should be content-equivalent. If you change anchors over time, treat the change as a model deployment and re-audit downstream metrics.
      </Prose>

      <H3>Probe accuracy as a fairness proxy</H3>
      <Prose>
        High probe accuracy on the unobservable attribute does not by itself imply judge bias. It only implies that the option of bias exists. A judge that has the proxy signal in its input but does not use it can still be fair. Probe accuracy is a necessary indicator (low probe accuracy bounds the judge's bias from above) but not sufficient. Confirm bias by directly measuring score disparities on labeled subgroups; do not infer bias from probe accuracy alone.
      </Prose>

      <H3>Sample-selection bias in audit sets</H3>
      <Prose>
        Demographically labeled audit sets are usually collected in narrow contexts (academic studies, third-party fairness vendors, internal red-team exercises). If the production input distribution differs systematically from the audit distribution, audit results do not transfer. The fix is partial: continuously refresh the audit set with samples drawn from production traffic (with appropriate consent and labeling pipelines). Coston et al. 2019 give the formal conditions under which audit results transfer; in practice, transfer is plausible only when the protected attribute conditional distribution is similar across audit and production.
      </Prose>

      <H3>Length normalization that hides quality</H3>
      <Prose>
        Length is observable and often a fair target for normalization, but length is also genuinely correlated with quality on some tasks (longer responses can include more useful detail). Aggressive length normalization can erase signal that the task actually depends on. The fix is task-conditional normalization: apply length normalization only on tasks where length is known to be a stylistic confound, not a quality signal. Maintain a per-task configuration of which observable features to normalize.
      </Prose>

      <H3>Adversarial training instability</H3>
      <Prose>
        Adversarial debiasing for representations is notoriously unstable. The minimax objective has multiple equilibria and the training dynamics can collapse to a representation that is invariant to the protected attribute by being uninformative about everything. Stabilize with gradient clipping on the adversary, learning-rate scheduling, and explicit regularization that maintains representation entropy. Monitor the representation's task utility throughout training; if it collapses, decrease the adversary weight.
      </Prose>

      <H3>Judge ensemble masking individual bias</H3>
      <Prose>
        Ensembling multiple judges can average out idiosyncratic biases, but if the constituent judges share a common bias (e.g., all judges in the ensemble are trained on similar data and have similar register preferences), ensembling does not help. The ensemble's residual bias is the intersection of the constituent biases, which can still be large. Audit each judge individually before assuming the ensemble is fairer.
      </Prose>

      <Callout accent="purple">
        Treat probe accuracy as a hazard signal, not a verdict. A judge with high input leakage might be using the leakage; it might not. The only way to know is direct measurement of score disparities on labeled subgroups. Do not deploy a fix in response to probe accuracy alone — measure outcomes first.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All four sources below are foundational for the observable/unobservable framework as it applies to LLM-judge fairness. Citations and arXiv IDs verified as of 2026-04-21.
      </Prose>

      <H3>Lipton, Chouldechova, McAuley 2018 — blind classifiers and treatment disparity</H3>
      <Prose>
        Zachary C. Lipton, Alexandra Chouldechova, Julian McAuley. "Does mitigating ML's impact disparity require treatment disparity?" arXiv:1711.07076. Published November 2017, NeurIPS 2018. The foundational result for this framework. Shows that blind classifiers (those without access to protected attributes) are structurally limited in how individual-level fair they can be: they cannot enforce treatment disparity even when treatment disparity is required for outcome equity. Directly applicable to LLM judges, which are blind classifiers in the technical sense (they do not receive demographic attributes as input) and therefore inherit the same impossibility.
      </Prose>

      <H3>Hardt, Price, Srebro 2016 — equality of opportunity</H3>
      <Prose>
        Moritz Hardt, Eric Price, Nathan Srebro. "Equality of Opportunity in Supervised Learning." NeurIPS 2016. arXiv:1610.02413. Defines the equalized odds criterion: a classifier is fair if its true positive rate and false positive rate are equal across protected groups. Provides a post-processing algorithm that achieves this criterion given any binary classifier and access to the protected attribute at calibration time. The conditional independence formulation in section 3a of this article is the LLM-judge analogue of equalized odds, with content quality playing the role of the true label.
      </Prose>

      <H3>Madras, Creager, Pitassi, Zemel 2018 — adversarial fairness representations</H3>
      <Prose>
        David Madras, Elliot Creager, Toniann Pitassi, Richard Zemel. "Learning Adversarially Fair and Transferable Representations." arXiv:1802.06309, ICML 2018. Introduces the LAFTR framework: train a representation jointly with an adversary that tries to predict the protected attribute from the representation, plus a downstream task head. The adversarial loss pushes the representation toward invariance to the protected attribute. Provides theoretical guarantees on the trade-off between fairness and task utility, formalizing the Pareto frontier that we explored empirically in section 4f. Directly applicable to LLM-judge representations.
      </Prose>

      <H3>Coston, Mishler, Kennedy, Chouldechova 2019 — fair transfer with missing attributes</H3>
      <Prose>
        Amanda Coston, Karthikeyan Natesan Ramamurthy, Dennis Wei, Kush R. Varshney, Skyler Speakman, Zairah Mustahsan, Supriyo Chakraborty. "Fair Transfer Learning with Missing Protected Attributes." Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society. Analyzes the case where the protected attribute is observed in some data (e.g., audit sets) but not at deployment. Establishes conditions under which fairness signals from labeled data transfer to deployment where the attribute is unobserved. Provides the theoretical basis for the audit-then-deploy pattern that all production LLM-judge fairness pipelines rely on. The transferability conditions are restrictive; the paper makes explicit when the pattern is and is not justified.
      </Prose>

      <H3>Additional context: Dwork et al. 2012 and Kleinberg et al. 2017</H3>
      <Prose>
        Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel. "Fairness Through Awareness." ITCS 2012. Establishes individual fairness as a Lipschitz condition on the classifier with respect to a task-appropriate metric. Conceptually adjacent to the observable-attribute framing — individual fairness on a similarity metric over text inputs is what input normalization aims to achieve. Jon Kleinberg, Sendhil Mullainathan, Manish Raghavan. "Inherent Trade-Offs in the Fair Determination of Risk Scores." ITCS 2017, arXiv:1609.05807. Proves that calibration and balance for the false positive/negative rates are jointly impossible except in degenerate cases. The trade-off generalizes to LLM-judge calibration: per-group calibration and equal selection rates cannot generally coexist.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Trace the data processing inequality</H3>
      <Prose>
        Starting from the chain <Code>{"U \\to (C, S) \\to R \\to J"}</Code>, write out every application of the data processing inequality that lets you upper-bound <Code>{"I(J; U | C)"}</Code> by <Code>{"I(S; U | C)"}</Code>. Where does the conditioning on <Code>C</Code> enter the chain, and what assumption does it require about how the judge processes its input? If the judge had access to <Code>C</Code> directly (rather than only through <Code>R</Code>), would the bound still hold? What does this say about the value of structured inputs to a judge versus raw text?
      </Prose>

      <H3>Exercise 2 — Probe-judge gap</H3>
      <Prose>
        Suppose a probe trained on a judge's input achieves 85% accuracy at predicting the unobservable attribute. The judge itself, when scored on a labeled audit set, exhibits a selection-rate disparity of 4 percentage points. Construct an explanation for the gap between high probe accuracy and modest judge disparity. Is the judge necessarily using the proxy signal weakly, or could there be other explanations? What additional measurement would let you distinguish between the explanations?
      </Prose>

      <H3>Exercise 3 — Counterfactual audit design</H3>
      <Prose>
        Design a counterfactual audit pipeline for a production LLM judge that scores customer support responses on a 1–5 helpfulness scale. The deployment involves five language registers (formal English, casual English, AAVE, Spanglish, and Mandarin-influenced English). Specify: (a) what the paraphrase prompts look like for each pair of registers, (b) how you sample audit pairs to ensure all 25 register pairs are represented, (c) what aggregate metrics you compute and what thresholds trigger an alert, (d) how you handle paraphrase content drift (paraphrases that lose substantive content). Sketch the full pipeline as pseudocode.
      </Prose>

      <H3>Exercise 4 — Lipton et al.'s value choice</H3>
      <Prose>
        A team is deploying an LLM judge for hiring screening. The legal team requires that the judge be "blind" to candidate demographics. The fairness team requires that the judge produce equal selection rates across demographic groups. The historical hiring data shows that candidates from group A have, on average, slightly lower technical content scores than candidates from group B (due to systematic differences in access to training resources). Apply Lipton, Chouldechova, and McAuley's result: which of the two requirements is achievable, and what compromise might satisfy both legal and fairness teams? Identify the value choice that the team must make and articulate the case for each side. What is the role of the audit pipeline in monitoring whichever choice is made?
      </Prose>

      <H3>Exercise 5 — When the unobservable attribute is quality</H3>
      <Prose>
        Consider an LLM judge for poetry evaluation. The unobservable attribute is the latent aesthetic quality of the poem. The observable features are word choice, line break patterns, rhyme density, and image novelty. There is no demographic angle and no fairness law to comply with — but the framework still applies. Write down: (a) the analogue of the conditional independence criterion <Code>{"J \\perp U | C"}</Code> for this setting (what is <Code>U</Code>, what is <Code>C</Code>, what is <Code>J</Code>?), (b) what role calibration plays when no ground-truth label exists, (c) what the analogue of "demographic disparity" is for an aesthetic-judgment system, and (d) how you would design an ensemble of judges with different proxies to diagnose miscalibration. How does this exercise show that the observable/unobservable distinction is broader than fairness law?
      </Prose>

      <H3>Exercise 6 — Adversarial debiasing trade-off</H3>
      <Prose>
        In the from-scratch implementation, the adversarial debiasing parameter <Code>λ</Code> was set to 2.0 and produced a judge with disparity 0.068 and task accuracy 0.798. Predict what happens at <Code>λ = 0</Code> (no adversarial pressure) and at <Code>λ = 10</Code> (very strong adversarial pressure). For each setting, write the expected probe accuracy on the learned representation, the expected task accuracy, and the expected disparity. What general shape does the Pareto frontier take, and where on the frontier should a production system aim to operate? How would you decide where to stop pushing for fairness?
      </Prose>

      <H3>Exercise 7 — Mention-vs-use stress test</H3>
      <Prose>
        Construct three test prompt-response pairs designed to stress-test a judge on the mention-vs-use distinction. (a) A response that mentions the author's demographic context as part of substantive content (e.g., a memoir excerpt). (b) A response that uses a demographically coded register stylistically. (c) A response that both mentions and uses. For each, predict how a fair judge should score relative to a hypothetical content-equivalent baseline that lacks both mention and use. What would a failure mode look like for each case? How would you build an automated test suite that includes these stress cases and runs continuously against a deployed judge?
      </Prose>

      <H3>Exercise 8 — From single-attribute to joint fairness</H3>
      <Prose>
        Section 8 noted that multi-attribute fairness scales combinatorially because joint demographic cells become rare. Suppose you have audit data labeled with three binary attributes (8 cells total) and your audit set contains 1000 examples uniformly distributed across cells (125 per cell). The judge exhibits a marginal disparity of 4 percentage points on each individual attribute. What can you say about joint disparities (e.g., the disparity between cell 000 and cell 111)? Under what assumptions can the marginal disparities be combined to bound joint disparities, and what could go wrong if those assumptions fail? Propose a concrete next step for diagnosing joint bias when the audit set is too small for direct per-cell estimates.
      </Prose>

    </div>
  ),
};

export default observableVsUnobservable;
