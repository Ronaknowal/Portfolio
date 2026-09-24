const guidanceByLevel = {
  foundation: "Build a clear explanation in your own words, keeping the conditions and limits that make it accurate.",
  intermediate: "Connect this idea to a model or workflow you already know, then test where the connection stops holding.",
  advanced: "Focus on assumptions, trade-offs, and the situations where a simpler method would be the better choice.",
  frontier: "Treat this as a research map: separate established ideas from active claims and keep the uncertainty visible.",
};

export default function LessonGuide({ topic }) {
  return (
    <aside className="lesson-guide" aria-label="How to approach this lesson">
      <div className="lesson-guide__label">LEARNING COMPASS</div>
      <div className="lesson-guide__body">
        <p className="lesson-guide__lead">You do not need to understand every line on the first read.</p>
        <p>{guidanceByLevel[topic.level]}</p>
      </div>
      <ol className="lesson-guide__steps">
        <li><strong>Orient.</strong> Start with the problem and the core intuition; skim historical detail until you have a mental picture.</li>
        <li><strong>Work through one example.</strong> Follow a diagram, equation, or code trace slowly enough to predict the next step.</li>
        <li><strong>Recall.</strong> Close the page and explain what the method does, when to use it, and one way it can fail.</li>
      </ol>
      {topic.intuitionAnchor && <a className="lesson-guide__jump" href={`#${topic.intuitionAnchor}`}>Start with the intuition →</a>}
    </aside>
  );
}
