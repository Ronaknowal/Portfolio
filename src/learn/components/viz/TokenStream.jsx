import { colors, fonts } from "../../styles";

// tokens: Array<string | { label: string, color?: string, title?: string }>
// highlight: number | null  — index of token to emphasize
export default function TokenStream({ tokens, highlight = null, label }) {
  const palette = [colors.gold, colors.green, "#c084fc", "#60a5fa", "#f472b6", "#facc15"];
  const normalize = (t, i) => typeof t === "string"
    ? { label: t, color: palette[i % palette.length] }
    : { color: palette[i % palette.length], ...t };

  return (
    <div style={{ margin: "16px 0" }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 6, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {tokens.map((t, i) => {
          const { label: txt, color, title } = normalize(t, i);
          const isHi = highlight === i;
          return (
            <span
              key={i}
              title={title || txt}
              style={{
                fontFamily: fonts.mono,
                fontSize: 12,
                padding: "3px 8px",
                borderRadius: 3,
                background: `${color}${isHi ? "44" : "22"}`,
                border: `1px solid ${color}${isHi ? "cc" : "55"}`,
                color: isHi ? "#fff" : color,
                whiteSpace: "pre",
                transition: "all 0.15s ease",
              }}
            >
              {txt}
            </span>
          );
        })}
      </div>
    </div>
  );
}
