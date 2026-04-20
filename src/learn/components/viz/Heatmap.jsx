import { colors, fonts } from "../../styles";

// matrix: number[][]   — rows × cols of values
// rowLabels, colLabels: string[]
// cellSize: px
export default function Heatmap({ matrix, rowLabels = [], colLabels = [], cellSize = 36, label, colorScale = "gold" }) {
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;

  const hues = {
    gold: (v) => `rgba(226,181,90,${v})`,
    green: (v) => `rgba(74,222,128,${v})`,
    purple: (v) => `rgba(192,132,252,${v})`,
  };
  const toColor = hues[colorScale] || hues.gold;

  return (
    <div style={{ margin: "16px 0", overflowX: "auto" }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 8, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}
      <table style={{ borderCollapse: "collapse", fontFamily: fonts.mono, fontSize: 11 }}>
        {colLabels.length > 0 && (
          <thead>
            <tr>
              <th />
              {colLabels.map((l) => (
                <th key={l} style={{ padding: 4, color: colors.textMuted, fontWeight: 400, writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 10 }}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {matrix.map((row, r) => (
            <tr key={r}>
              {rowLabels[r] !== undefined && (
                <td style={{ padding: "0 8px 0 0", color: colors.textMuted, textAlign: "right", fontSize: 10 }}>
                  {rowLabels[r]}
                </td>
              )}
              {row.map((v, c) => {
                const norm = (v - min) / range;
                return (
                  <td
                    key={c}
                    title={`${rowLabels[r] || ""} × ${colLabels[c] || ""}: ${v}`}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      background: toColor(0.1 + norm * 0.8),
                      border: `1px solid ${colors.bg}`,
                      color: norm > 0.6 ? "#000" : colors.textPrimary,
                      textAlign: "center",
                      fontSize: 10,
                    }}
                  >
                    {Number.isInteger(v) ? v : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
