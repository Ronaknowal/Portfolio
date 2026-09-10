import { Callout, CodeBlock, H2, H3, Prose } from "./content";

export default function ProgrammingReference({ overview, model, groups, example, pitfalls, practice }) {
  return <div>
    <H2>1. What this tool is for</H2><Prose>{overview}</Prose>
    <H2>2. The mental model</H2><Prose>{model}</Prose>
    <H2>3. Practical API map</H2>
    <Prose>Use this as a working reference. Each group starts collapsed so you can learn the family first, then open the functions you need.</Prose>
    <div className="programming-reference">
      {groups.map((group) => <details key={group.name}>
        <summary><span>{group.name}</span><span>{group.use}</span></summary>
        <div className="programming-reference__functions">
          {group.items.map((item) => <div key={item.name}><code>{item.name}</code><p>{item.description}</p></div>)}
        </div>
      </details>)}
    </div>
    <H2>4. A worked pattern</H2>
    <CodeBlock language={example.language || "python"}>{example.code}</CodeBlock>
    <Callout accent="green" label="Why this works">{example.explanation}</Callout>
    <H2>5. Debugging and safe usage</H2><Prose>{pitfalls}</Prose>
    <H2>6. Practice</H2><Prose>{practice}</Prose>
  </div>;
}
