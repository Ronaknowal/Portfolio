import { fonts, levelColors, levelLabels } from "../styles";

export default function LevelBadge({ level, size = "small" }) {
  const color = levelColors[level] || "#555";
  const label = levelLabels[level] || level;
  const isSmall = size === "small";

  return (
    <span
      style={{
        fontFamily: fonts.mono,
        fontSize: isSmall ? 7 : 9,
        color,
        padding: isSmall ? "1px 4px" : "2px 6px",
        border: `1px solid ${color}33`,
        borderRadius: 2,
        textTransform: "uppercase",
        letterSpacing: 0.5,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}
