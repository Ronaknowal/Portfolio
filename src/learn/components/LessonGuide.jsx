import { useLayoutEffect, useState } from "react";
import { collectLessonSections } from "./lesson-navigation.js";
import "./lesson-guide.css";

export default function LessonGuide({ articleRef, content, topic, prerequisiteTopics, isComplete, slots }) {
  const [sections, setSections] = useState([]);
  useLayoutEffect(() => {
    if (articleRef.current) setSections(collectLessonSections(articleRef.current));
  }, [articleRef, content]);

  function focusSection(event, id) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    // The anchor owns URL/history/scrolling. Focus follows it for keyboard use.
    articleRef.current?.ownerDocument.getElementById(id)?.focus({ preventScroll: true });
  }

  return <aside className="lesson-compass" aria-label="Learning compass">
    <div className="lesson-compass__heading">
      <p className="lesson-compass__eyebrow">Learning compass</p>
      {sections.length > 0 && <span>{sections.length} sections</span>}
    </div>
    <div ref={slots.summary} className="lesson-compass__summary" />
    <p className="lesson-compass__fallback">Follow the lesson in order, or jump to the part you want to revisit.</p>
    <div ref={slots.route} className="lesson-compass__route" />
    <details className="lesson-compass__preparation">
      <summary>Before you start</summary>
      <div ref={slots.prerequisites} className="lesson-compass__prerequisites" />
      {prerequisiteTopics.length > 0 ? <>
        <p>Review these supporting topics if their ideas are unfamiliar.</p>
        <ul>{prerequisiteTopics.map(item => <li key={item.id}>
          <a href={`/learn/topic/${item.id}`}>{item.title}</a>{isComplete(item.id) ? " · completed" : ""}
        </li>)}</ul>
      </> : <p className="lesson-compass__prerequisites-fallback">The lesson introduces its prerequisites where they are needed.</p>}
    </details>
    <details className="lesson-compass__exploration">
      <summary>Using the examples and labs</summary>
      <div ref={slots.exploration} className="lesson-compass__exploration-notes" />
      <p className="lesson-compass__exploration-fallback">Change the available controls and follow their effects. Work through the examples, then use the practice to check your understanding.</p>
    </details>
    {sections.length > 0 && <nav aria-label={`Sections in ${topic.title}`}>
      <p className="lesson-compass__index-label">In this lesson</p>
      <ul className="lesson-compass__sections">
        {sections.map(section => <li key={section.id}>
          <a href={`#${section.id}`} onClick={event => focusSection(event, section.id)}>
            <span className="lesson-compass__section-number" aria-hidden={section.number ? undefined : true}>{section.number ? `${section.number}.` : "—"}</span>
            <span>{section.title}</span>
          </a>
        </li>)}
      </ul>
    </nav>}
  </aside>;
}
