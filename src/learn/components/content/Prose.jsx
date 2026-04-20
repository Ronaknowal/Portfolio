import { colors, fonts } from "../../styles";

export function Prose({ children, dim = false }) {
  return (
    <p style={{
      fontFamily: fonts.mono,
      fontSize: 13,
      color: dim ? colors.textMuted : colors.textSecondary,
      lineHeight: 1.8,
      marginBottom: 16,
    }}>
      {children}
    </p>
  );
}
