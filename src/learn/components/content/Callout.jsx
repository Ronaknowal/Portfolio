import { colors, fonts } from "../../styles";

export function Callout({ children, accent = "gold" }) {
  const accentColor = accent === "green" ? colors.green
    : accent === "red" ? "#f87171"
    : colors.gold;
  return (
    <div style={{
      borderLeft: `2px solid ${accentColor}`,
      padding: "4px 0 4px 16px",
      margin: "16px 0",
      fontFamily: fonts.mono,
      fontSize: 12,
      color: colors.textSecondary,
      lineHeight: 1.7,
    }}>
      {children}
    </div>
  );
}
