import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const rlvr = {
  title: "RLVR (Reinforcement Learning with Verifiable Rewards)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        The reward model at the center of RLHF is a neural network — a learned, imperfect proxy for human preferences, and the first place reward hacking enters the pipeline. As covered in the RLHF topic, a powerful optimizer will eventually find inputs where the proxy diverges from what it was supposed to represent: the reward score climbs while actual human preference falls. This is not a bug in any particular implementation. It is the consequence of optimizing against an approximation rather than the thing itself.
      </Prose>

      <Prose>
        For a specific class of tasks — math problems, code generation, formal proofs, logic puzzles, anything with a mechanically checkable answer — the reward model is unnecessary. The correct answer is either present or it is not, and a deterministic function can tell you which without consulting a neural network. RLVR replaces the learned reward model with a verifier: a program that returns 1 for a correct answer and 0 for a wrong one. No parameters to overfit, no proxy to hack, no calibration to worry about. The signal is exact by construction.
      </Prose>

      <H2>Where the verifier comes from</H2>

      <Prose>
        The verifier is task-specific, but the patterns are narrow. For GSM8K-style arithmetic and competition math, the verifier parses the final boxed answer from the model's output and compares it numerically to the ground truth. For code generation, it compiles the generated program and runs it against a hidden test suite, returning the fraction of tests that pass. For formal theorem proving — Lean, Isabelle, Coq — it submits the proof to the type-checker and returns whether it type-checks. In each case the verifier is a small, fast, interpretable function; the hard work of specifying what "correct" means was already done when someone wrote the problem and its ground-truth answer.
      </Prose>

      <CodeBlock language="python">
{`def math_verifier(response: str, ground_truth: float) -> float:
    """Extracts a numeric final answer and checks equality. Reward is 0/1."""
    import re
    match = re.search(r"\\\\boxed\\{([^}]+)\\}", response)
    if not match: return 0.0
    try:
        return float(abs(float(match.group(1)) - ground_truth) < 1e-6)
    except ValueError:
        return 0.0

def code_verifier(code: str, test_cases: list) -> float:
    """Runs code against test cases in a sandbox. Reward is fraction passing."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        passed = sum(tc(exec_globals) for tc in test_cases)
        return passed / len(test_cases)
    except Exception:
        return 0.0`}
      </CodeBlock>

      <Prose>
        The math verifier above is essentially a regex and a float comparison — thirty lines to replace a seven-billion-parameter reward model for an entire domain. The code verifier is slightly more elaborate, but the concept is the same: correctness is a function you can compute, so compute it. The sandboxing requirements for code execution are real engineering work, but the reward function itself is trivial.
      </Prose>

      <H2>Why this is a big deal</H2>

      <Prose>
        Three things a learned reward model does poorly, a verifier does perfectly. First, the signal is exact. There is no calibration drift, no Bradley-Terry approximation, no ambiguity in what the reward means. A verifier does not become less accurate as the policy improves; it does not produce higher scores for long responses or responses with confident tone. It says whether the answer is right. Second, the signal scales without labels. A math dataset with a hundred thousand problems and their answers supports a hundred thousand verifier queries per training step, none of which required a human labeler. You can generate synthetic math problems at scale — and the RLVR pipeline will score them all for free. Third, the signal is stable. The verifier's behavior does not change as the policy changes. A reward model trained on the initial policy's outputs will become increasingly misaligned as the policy moves; a verifier checking against the ground truth is equally valid at step 1 and step 100,000.
      </Prose>

      <Prose>
        The practical consequence is that RLVR training can continue far longer than RLHF before hitting the reward-hacking ceiling. With a learned reward model, the policy eventually finds the cracks in the proxy and exploits them; training past that point actively degrades quality. With a verifier, there are no cracks to find. The policy can only improve its reward score by producing more correct answers. This is what makes RLVR the right tool for the tasks it fits: the training signal stays honest regardless of how hard the optimizer pushes.
      </Prose>

      <H3>The cost</H3>

      <Prose>
        RLVR works only for problems where correctness can be mechanized, and that set is narrower than it first appears. "Write a clear explanation" is not verifiable. "Prove this theorem" is. Arithmetic at the grade-school level is verifiable by regex and float comparison; arithmetic at the research level may require a formal proof assistant. Code with a complete unit test suite is verifiable; code described only as "make this UI feel better" is not. The boundary between verifiable and non-verifiable roughly coincides with the boundary between symbolic and aesthetic tasks. Open-ended generation, translation quality, tone, helpfulness in conversation — none of these have a mechanical ground truth, and RLVR offers nothing for them. For those tasks, the preference-based pipeline described in the RLHF topic remains the only option. RLVR does not replace RLHF; it bypasses it for the subset of tasks where the bypass is possible.
      </Prose>

      <H2>Exploration — the bottleneck</H2>

      <Prose>
        The hardest problem in RLVR is not the verifier — it is getting a reward signal at all early in training. The policy starts from a base or SFT checkpoint that cannot reliably solve the target problems. On any given problem, most rollouts return reward 0. A training step where every rollout returns 0 contributes no gradient: the policy cannot learn which responses are better than which other responses when all responses are equally wrong. If the problem set is too hard for the starting policy, training stalls entirely.
      </Prose>

      <Prose>
        The standard fix is curriculum: begin training on problems where the base model succeeds often enough to produce non-zero rewards, and gradually introduce harder problems as the policy improves. DeepSeek's GRPO-based RLVR uses group sampling — generating N=8 to 16 responses per prompt instead of one — specifically to increase the probability that at least some rollouts succeed. The advantage is computed within each group: a response that gets reward 1 is compared to responses that got reward 0, and the policy is updated toward the winner. Without group sampling on a hard problem set, the expected gradient is near zero; with it, even a 1-in-8 success rate produces a clean learning signal. This is the central algorithmic contribution of GRPO-style training, described in detail in the GRPO/RLOO/KTO topic.
      </Prose>

      <H3>Synthetic data amplification</H3>

      <Prose>
        A common pattern in deployed RLVR pipelines extends the basic loop with a synthetic data step. After each round of RLVR, the policy has produced some rollouts the verifier accepted. Those correct rollouts — full reasoning chains ending in verified answers — are used as supervised fine-tuning data for the next round. The intuition is that correct rollouts demonstrate reasoning patterns the model had not reliably produced before; SFT on them shifts the base distribution toward those patterns before the next RL phase begins. This bootstrapping is what allows RLVR to reach hard problems: the policy does not have to rediscover good reasoning from scratch each round, it starts each round slightly better than it ended the last.
      </Prose>

      <StepTrace
        label="rlvr with synthetic-data amplification"
        steps={[
          { label: "1. sample rollouts", render: () => (
            <TokenStream tokens={["policy", " →", " N rollouts per problem", " →", " keep correct ones"]} />
          ) },
          { label: "2. SFT on winners", render: () => (
            <TokenStream tokens={["correct rollouts", " →", " supervised fine-tune", " →", " stronger policy"]} />
          ) },
          { label: "3. rlvr on all problems", render: () => (
            <TokenStream tokens={["stronger policy", " →", " GRPO with verifier", " →", " better policy"]} />
          ) },
          { label: "4. repeat", render: () => (
            <TokenStream tokens={["loop ×N iterations"]} />
          ) },
        ]}
      />

      <H2>Where RLVR is winning now</H2>

      <Prose>
        The benchmarks where RLVR has produced the clearest gains are the ones with exact verifiers: competition math (AIME, the full MATH benchmark), competitive programming (Codeforces problems scored by a judge), formal theorem proving (miniF2F, evaluated by Lean), and long-horizon agentic tasks where success is externally checkable — "resolve this GitHub issue and pass the repository's test suite" is verifiable in exactly the way "write a helpful response" is not. DeepSeek-R1 used RLVR as the primary post-training signal for its reasoning capabilities, with group sampling and curriculum as the core mechanisms. The near-superhuman performance of the o-series models on verifiable benchmarks is, by the available evidence, primarily an RLVR story rather than an RLHF one. The reward model was removed from the loop for the tasks where it could be removed, and the results were better precisely because the training signal was honest.
      </Prose>

      <Prose>
        The trajectory is clear. For any domain that can be equipped with a verifier — which increasingly includes agentic computer-use tasks evaluated by whether the computer ends up in a target state — RLVR is the default post-training method. The investment in building the verifier and the test suite is paid back in training signal quality throughout the entire training run.
      </Prose>

      <H3>The generalization question</H3>

      <Prose>
        The open question is whether RLVR on verifiable domains transfers to non-verifiable reasoning. The available evidence suggests it does. Models RL-trained on math and code tend to improve on general reasoning benchmarks, chain-of-thought structure, and instruction following — tasks where no verifier was present during training. The informal account is that RLVR teaches the model a reasoning discipline: produce a chain of steps that leads to a conclusion you can actually verify, rather than a chain that sounds plausible. That discipline appears to transfer. The model learns to think in a way that produces correct outputs on verifiable tasks, and the style of thinking carries over to tasks where correctness is harder to define.
      </Prose>

      <Callout accent="gold">
        Verifiable rewards aren't just a trick for narrow tasks — they seem to produce reasoning behavior that transfers. That transfer is the real find of the RLVR era.
      </Callout>

      <Prose>
        The mechanism is not fully understood. It could be that systematic chain-of-thought structure is simply useful everywhere, and RLVR selects for it because unstructured responses are less likely to land on verifiably correct answers. It could be that the training signal's honesty during RL allows the model to update on real reasoning errors rather than proxy artifacts, producing more deeply learned habits. Whatever the mechanism, the empirical pattern from DeepSeek-R1 and similar models is consistent: RLVR on math and code does not confine its effects to math and code.
      </Prose>

      <H2>Closing</H2>

      <Prose>
        RLVR is what post-training looks like when you remove the learned reward model entirely. The machinery is simpler: a verifier replaces a seven-billion-parameter network, group sampling handles the exploration problem, and the training loop is otherwise standard policy gradient. What it asks for in return is a domain where correctness is a function you can compute — and for the growing set of domains that qualify, from competition math to formal proofs to agentic software engineering tasks, it has become the quiet default. RLHF remains necessary for everything else. But the boundary of "everything else" is narrower than it was two years ago, and it keeps moving.
      </Prose>
    </div>
  ),
};

export default rlvr;
