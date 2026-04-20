import { colors, fonts } from "../../styles";

export function Code({ children }) {
  return (
    <code style={{
      fontFamily: fonts.mono,
      fontSize: 12,
      color: colors.gold,
      background: "rgba(226,181,90,0.08)",
      padding: "1px 5px",
      borderRadius: 3,
    }}>
      {children}
    </code>
  );
}

export function CodeBlock({ children, language }) {
  return (
    <div style={{
      background: "rgba(0,0,0,0.4)",
      border: `1px solid ${colors.border}`,
      borderRadius: 4,
      padding: 16,
      fontFamily: fonts.mono,
      fontSize: 12,
      color: colors.textSecondary,
      lineHeight: 1.7,
      marginBottom: 16,
      overflowX: "auto",
      whiteSpace: "pre",
    }}>
      {language && (
        <div style={{ color: colors.textDark, fontSize: 10, marginBottom: 8, letterSpacing: 1 }}>
          {language.toUpperCase()}
        </div>
      )}
      {children}
    </div>
  );
}
