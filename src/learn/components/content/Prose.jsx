import { colors, fonts } from "../../styles";

export function Prose({ children, dim = false }) {
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
