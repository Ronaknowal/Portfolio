import { useState, useMemo } from "react";
import { colors, fonts } from "../styles";

// Deterministic pseudo-random based on string hash — gives consistent "embeddings"
function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return hash;
}

function pseudoEmbedding(token, dim = 8) {
  const vec = [];
  for (let i = 0; i < dim; i++) {
    const seed = hashCode(token + "_" + i);
    vec.push(Math.sin(seed * 0.01) * 0.5 + Math.cos(seed * 0.007) * 0.5);
  }
  return vec;
}

function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function computeAttention(tokens) {
  const dim = 8;
  const embeddings = tokens.map((t) => pseudoEmbedding(t, dim));
  const scale = Math.sqrt(dim);

  // attention[i][j] = softmax(Q_i . K_j / sqrt(d))
  return tokens.map((_, i) => {
    const scores = tokens.map((_, j) => dotProduct(embeddings[i], embeddings[j]) / scale);
    return softmax(scores);
  });
}

const PRESETS = [
  { label: "Translation", tokens: ["The", "cat", "sat", "on", "the", "mat"] },
  { label: "Coreference", tokens: ["Alice", "said", "she", "would", "go", "home"] },
  { label: "Negation", tokens: ["I", "do", "not", "like", "green", "eggs"] },
];

export default function AttentionVisualizer() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [hoveredCell, setHoveredCell] = useState(null); // { row, col }

  const tokens = PRESETS[presetIdx].tokens;
  const weights = useMemo(() => computeAttention(tokens), [tokens]);

  const cellSize = 48;
  const labelWidth = 64;
  const n = tokens.length;

  // Color interpolation: transparent → gold based on weight
  const weightToColor = (w) => {
    const alpha = Math.max(0.05, Math.min(w, 1));
    return `rgba(226, 181, 90, ${alpha})`;
  };

  return (
    <div style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: 20,
      background: "rgba(0,0,0,0.3)",
      marginBottom: 20,
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.gold, letterSpacing: 1, marginBottom: 4 }}>
            INTERACTIVE
          </div>
          <div style={{ fontFamily: fonts.sans, fontSize: 14, fontWeight: 500, color: colors.textPrimary }}>
            Attention Weight Visualizer
          </div>
        </div>
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim }}>
          hover cells to inspect weights
        </div>
      </div>

      {/* Preset selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 20 }}>
        {PRESETS.map((p, i) => (
          <span
            key={i}
            onClick={() => { setPresetIdx(i); setHoveredCell(null); }}
            style={{
              padding: "4px 10px",
              borderRadius: 3,
              fontFamily: fonts.mono,
              fontSize: 9,
              color: i === presetIdx ? colors.gold : colors.textMuted,
              border: `1px solid ${i === presetIdx ? `${colors.gold}44` : colors.border}`,
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
          >
            {p.label}
          </span>
        ))}
      </div>

      {/* Heatmap */}
      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "inline-block" }}>
          {/* Column headers (Keys) */}
          <div style={{ display: "flex", marginLeft: labelWidth }}>
            {tokens.map((t, j) => (
              <div
                key={j}
                style={{
                  width: cellSize,
                  textAlign: "center",
                  fontFamily: fonts.mono,
                  fontSize: 10,
                  color: hoveredCell && hoveredCell.col === j ? colors.gold : colors.textMuted,
                  paddingBottom: 6,
                  transition: "color 0.15s ease",
                }}
              >
                {t}
              </div>
            ))}
          </div>

          {/* Label row: "Keys →" */}
          <div style={{
            marginLeft: labelWidth,
            fontFamily: fonts.mono,
            fontSize: 8,
            color: colors.textDark,
            letterSpacing: 1,
            marginBottom: 4,
            textAlign: "center",
            width: cellSize * n,
          }}>
            KEYS →
          </div>

          {/* Rows */}
          {tokens.map((rowToken, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              {/* Row label (Query) */}
              <div style={{
                width: labelWidth,
                textAlign: "right",
                paddingRight: 8,
                fontFamily: fonts.mono,
                fontSize: 10,
                color: hoveredCell && hoveredCell.row === i ? colors.gold : colors.textMuted,
                transition: "color 0.15s ease",
              }}>
                {rowToken}
              </div>

              {/* Cells */}
              {weights[i].map((w, j) => {
                const isHovered = hoveredCell && hoveredCell.row === i && hoveredCell.col === j;
                const isRowHighlight = hoveredCell && hoveredCell.row === i;
                const isColHighlight = hoveredCell && hoveredCell.col === j;

                return (
                  <div
                    key={j}
                    onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                    onMouseLeave={() => setHoveredCell(null)}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background: weightToColor(w),
                      border: isHovered
                        ? `1px solid ${colors.gold}`
                        : `1px solid ${(isRowHighlight || isColHighlight) ? colors.border : "rgba(255,255,255,0.03)"}`,
                      fontFamily: fonts.mono,
                      fontSize: isHovered ? 11 : 9,
                      color: isHovered ? colors.textPrimary : w > 0.3 ? colors.textSecondary : colors.textDark,
                      cursor: "crosshair",
                      transition: "all 0.1s ease",
                    }}
                  >
                    {w.toFixed(2)}
                  </div>
                );
              })}
            </div>
          ))}

          {/* Row label: "QUERIES ↓" */}
          <div style={{
            width: labelWidth,
            textAlign: "right",
            paddingRight: 8,
            fontFamily: fonts.mono,
            fontSize: 8,
            color: colors.textDark,
            letterSpacing: 1,
            marginTop: 4,
          }}>
            QUERIES ↑
          </div>
        </div>
      </div>

      {/* Hover detail */}
      <div style={{
        marginTop: 14,
        padding: "8px 12px",
        background: "rgba(0,0,0,0.3)",
        borderRadius: 4,
        fontFamily: fonts.mono,
        fontSize: 10,
        color: colors.textDim,
        minHeight: 20,
      }}>
        {hoveredCell ? (
          <>
            <span style={{ color: colors.gold }}>Query:</span>{" "}
            <span style={{ color: colors.textSecondary }}>"{tokens[hoveredCell.row]}"</span>
            {"  →  "}
            <span style={{ color: colors.gold }}>Key:</span>{" "}
            <span style={{ color: colors.textSecondary }}>"{tokens[hoveredCell.col]}"</span>
            {"  =  "}
            <span style={{ color: colors.green }}>
              {weights[hoveredCell.row][hoveredCell.col].toFixed(4)}
            </span>
            {"  "}
            <span style={{ color: colors.textDark }}>
              (softmax of Q·K/√d)
            </span>
          </>
        ) : (
          <span>Hover over a cell to see the attention weight between query and key tokens</span>
        )}
      </div>
    </div>
  );
}
