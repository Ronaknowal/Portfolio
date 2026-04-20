import { colors, fonts } from "../../styles";

export function H2({ children }) {
  return (
    <h2 style={{
      fontFamily: fonts.sans,
      fontSize: 22,
      fontWeight: 600,
      color: colors.textPrimary,
      margin: "32px 0 14px",
    }}>
      {children}
    </h2>
  );
}

export function H3({ children }) {
  return (
    <h3 style={{
      fontFamily: fonts.sans,
      fontSize: 20,
      fontWeight: 600,
      color: colors.textPrimary,
      margin: "24px 0 12px",
    }}>
      {children}
    </h3>
  );
}
