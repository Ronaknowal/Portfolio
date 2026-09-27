import { colors, fonts } from "../../styles";
import { LessonOpeningNote } from "../LessonOpening.jsx";

export function Prose({ children, dim = false, opening }) {
  if (opening) return <LessonOpeningNote kind={opening}><p>{children}</p></LessonOpeningNote>;
  return (
    <p style={{
      fontFamily: fonts.sans,
      fontSize: "clamp(1rem, 1.25vw, 1.08rem)",
      color: dim ? colors.textMuted : "#bdbdbd",
      lineHeight: 1.82,
      letterSpacing: "-0.005em",
      marginBottom: 20,
    }}>
      {children}
    </p>
  );
}
