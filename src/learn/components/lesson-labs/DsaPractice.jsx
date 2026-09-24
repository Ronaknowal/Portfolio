import './dsa-practice.css';

function PracticeProblems({ problems }) {
  return <ol className="dsa-practice__problems">
    {problems.map(problem => <li key={problem.number} className="dsa-practice__problem">
      <div className="dsa-practice__problem-heading">
        <a href={`https://leetcode.com/problems/${problem.slug}/`} target="_blank" rel="noopener noreferrer">
          <span className="dsa-practice__number">{problem.number}.</span> {problem.title}
          <span className="dsa-practice__sr-only"> (opens in a new tab)</span>
        </a>
        <span className="dsa-practice__difficulty">{problem.difficulty}</span>
      </div>
      <p>{problem.focus}</p>
      {problem.prerequisite && <p className="dsa-practice__prerequisite"><strong>Before attempting:</strong> {problem.prerequisite}</p>}
      <div className="dsa-practice__reveals">
        <details>
          <summary>Optional hint<span className="dsa-practice__sr-only"> for {problem.title}</span></summary>
          <p>{problem.hint}</p>
        </details>
        <details>
          <summary>After solving: test transfer<span className="dsa-practice__sr-only"> for {problem.title}</span></summary>
          <p>{problem.transfer}</p>
        </details>
      </div>
    </li>)}
  </ol>;
}

/** Presentation only: each lesson imports and passes its own practice dataset. */
export function DsaPractice({ practice }) {
  return <section className="dsa-practice" aria-labelledby="guided-dsa-practice" data-practice-topic={practice.topicId}>
    <h2 id="guided-dsa-practice">Guided LeetCode practice</h2>
    <p>{practice.introduction}</p>
    <p className="dsa-practice__method">Start with the lesson's independent exercises. For each problem below, write the input/output contract, draw a small state trace, and propose a correct baseline before optimizing. Explain your invariant, time/space costs and edge cases before submitting. Open hints only when you need them.</p>
    <p className="dsa-practice__access">LeetCode's difficulty labels describe the platform's problems; the stages below describe this lesson's learning route. Problem links open in a new tab. Official statements were accessible when checked on {practice.verifiedOn}. Submission may require an account; editorials and other features can have separate access limits.</p>
    {practice.groups.map(group => group.optional
      ? <details className="dsa-practice__extension" key={group.id}>
          <summary><span>{group.title}</span><small>{group.problems.length} optional problems · check prerequisites</small></summary>
          <p>{group.introduction}</p>
          <PracticeProblems problems={group.problems}/>
        </details>
      : <div className="dsa-practice__stage" key={group.id}>
          <h3>{group.title}</h3>
          <p>{group.introduction}</p>
          <PracticeProblems problems={group.problems}/>
        </div>)}
    <div className="dsa-practice__readiness">
      <h3>Make the learning reusable</h3>
      <p>After studying a hint or solution, close it and rebuild the reasoning in a later session. Reattempt with changed inputs, then explain why an alternative fails. Mix an earlier topic into later practice so the structure is a choice you make.</p>
      <ul>{practice.readiness.map(item => <li key={item}>{item}</li>)}</ul>
      <p>{practice.localBridge}</p>
      <p>Use this set to build a broad, reusable toolkit. Completing a fixed list cannot guarantee that every future interview problem will be solvable; readiness is the ability to justify and adapt an approach to an unfamiliar constraint.</p>
    </div>
  </section>;
}
