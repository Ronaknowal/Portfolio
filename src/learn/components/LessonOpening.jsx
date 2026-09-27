import { createContext, useContext } from "react";
import { createPortal } from "react-dom";

// The reader owns placement; a lesson owns its specific learning guidance.
// Outside the reader (including standalone rendering), keep guidance inline.
export const LessonOpeningContext = createContext(null);

export function LessonOpeningNote({ children, kind = "route" }) {
  const targets = useContext(LessonOpeningContext);
  const note = <div className={`lesson-compass__note lesson-compass__note--${kind}`}>{children}</div>;
  if (targets === null) return note;
  return targets[kind] ? createPortal(note, targets[kind]) : null;
}

export function LessonOrientation({ children, prerequisites, exampleKind = "Python" }) {
  return <>
    {children && <LessonOpeningNote kind="summary"><p>{children}</p></LessonOpeningNote>}
    {prerequisites && <LessonOpeningNote kind="prerequisites"><p>{prerequisites}</p></LessonOpeningNote>}
    <LessonOpeningNote kind="exploration">
      <p>The interactive labs run on this page. {exampleKind} examples run in your own environment.</p>
    </LessonOpeningNote>
  </>;
}
