export const colors = {
  gold: "#e2b55a",
  green: "#4ade80",
  bg: "#050505",
  textPrimary: "#e8e8e8",
  textSecondary: "#888",
  textMuted: "#555",
  textDim: "#444",
  textDark: "#333",
  border: "#1a1a1a",
  cardBg: "rgba(255,255,255,0.02)",
  cardHoverBg: "rgba(226,181,90,0.04)",
  cardHoverBorder: "#e2b55a33",
};

// Difficulty level colors (single-hue approach — gold shades for consistency)
export const levelColors = {
  foundation: "#4ade80",   // green — basics
  intermediate: "#e2b55a", // gold — core
  advanced: "#c084fc",     // purple — deep
  frontier: "#f87171",     // red — cutting edge
};

export const levelLabels = {
  foundation: "Foundation",
  intermediate: "Intermediate",
  advanced: "Advanced",
  frontier: "Frontier",
};

export const fonts = {
  mono: "'JetBrains Mono', 'Fira Code', monospace",
  sans: "'Space Grotesk', sans-serif",
};

export const navLinkStyle = {
  fontFamily: fonts.mono,
  fontSize: 11,
  color: colors.textMuted,
  textDecoration: "none",
  letterSpacing: 1,
  textTransform: "uppercase",
  transition: "color 0.2s",
};

export const epochLabelStyle = {
  fontFamily: fonts.mono,
  fontSize: 10,
  color: colors.textDark,
  letterSpacing: 3,
  marginBottom: 40,
};

export const sectionTitleStyle = {
  fontFamily: fonts.sans,
  fontSize: 40,
  fontWeight: 600,
  letterSpacing: "-0.03em",
  margin: "0 0 12px 0",
  color: colors.textPrimary,
};

export const subtitleStyle = {
  fontFamily: fonts.mono,
  fontSize: 12,
  color: colors.textDim,
  marginBottom: 40,
};
