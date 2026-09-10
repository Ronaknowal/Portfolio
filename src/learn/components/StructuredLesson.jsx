import { Callout, CodeBlock, H2, H3, Prose } from "./content";
import { programmingExpectations, programmingReferences } from "../data/programming-reference-data";

// Legacy compact article helper. New teaching increments follow the current
// handoff and compose concept-specific sections, examples and representations.
// This component's shape is not the authoring standard or a lab/example quota.
export default function StructuredLesson({ why, intuition, concepts, example, practice, pitfalls, next, referenceKey }) {
  const groups = programmingReferences[referenceKey] || [];
  const expectation = programmingExpectations[referenceKey];
  return (
    <div>
      <H2>1. Why this matters</H2>
      <Prose>{why}</Prose>

      <H2>2. Build the intuition</H2>
      <Prose>{intuition}</Prose>

      <H2>3. Put it into practice</H2>
      {concepts.map((concept) => (
        <section key={concept.title}>
          <H3>{concept.title}</H3>
          <Prose>{concept.body}</Prose>
        </section>
      ))}
      {groups.length > 0 && <>
        <H3>Programming reference</H3>
        <div className="programming-reference">{groups.map((group) => <details key={group.name}>
          <summary><span>{group.name}</span><span>{group.use}</span></summary>
          <div className="programming-reference__functions">{group.items.map((item) => <div key={item.name}><code>{item.name}</code><p>{item.description}</p></div>)}</div>
        </details>)}</div>
      </>}
      <CodeBlock language={example.language || "python"}>{example.code}</CodeBlock>
      <Callout accent="green" label="Read the example">
        {example.explanation}
      </Callout>
      {expectation && <Callout label="What you should observe">{expectation}</Callout>}

      <H2>4. Check your understanding</H2>
      <Prose>{practice}</Prose>

      <H2>5. Know where it can fail</H2>
      <Prose>{pitfalls}</Prose>
      <Callout label="Next step">{next}</Callout>
    </div>
  );
}
