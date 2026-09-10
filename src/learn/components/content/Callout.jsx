import { colors, fonts } from "../../styles";

export function Callout({ children, accent = "gold", label, title, type }) {
  const accentColor = accent === "green" ? colors.green
    : accent === "red" ? "#f87171"
    : colors.gold;
  const calloutLabel = title || label || (type === "answer" ? "Worked answer" : "Pause & reflect");
  return (
    <aside role="note" aria-label={calloutLabel} style={{
      borderLeft: `2px solid ${accentColor}`,
      padding: "8px 0 8px 16px",
      margin: "20px 0",
      fontFamily: fonts.sans,
      fontSize: 14,
      color: "#bdbdbd",
      lineHeight: 1.7,
    }}>
      <div style={{
        marginBottom: 6,
        color: accentColor,
        fontFamily: fonts.mono,
        fontSize: 9,
        letterSpacing: 1,
        textTransform: "uppercase",
      }}>
        {calloutLabel}
      </div>
      {children}
    </aside>
  );
}
