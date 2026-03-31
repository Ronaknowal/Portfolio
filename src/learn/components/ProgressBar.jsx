import { colors } from "../styles";

export default function ProgressBar({ percent, height = 3, color = colors.green }) {
  return (
    <div
      style={{
        height,
        background: colors.border,
        borderRadius: height / 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.round(percent * 100)}%`,
          height: "100%",
          background: color,
          borderRadius: height / 2,
          transition: "width 0.3s ease",
        }}
      />
    </div>
  );
}
