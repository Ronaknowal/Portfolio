import { useState } from "react";
import { colors, fonts } from "../../styles";

// steps: Array<{ label: string, render: () => JSX }>
export default function StepTrace({ steps, label }) {
  const [i, setI] = useState(0);
  const step = steps[i];

  return (
    <div style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: 16,
      margin: "16px 0",
      background: colors.cardBg,
    }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 10, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}

      <div style={{ minHeight: 60, marginBottom: 12 }}>
        {step.render()}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <button
          onClick={() => setI((x) => Math.max(0, x - 1))}
          disabled={i === 0}
          style={btnStyle(i === 0)}
        >
          ← Prev
        </button>
        <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textMuted }}>
          Step {i + 1} / {steps.length} {step.label ? `· ${step.label}` : ""}
        </div>
        <button
          onClick={() => setI((x) => Math.min(steps.length - 1, x + 1))}
          disabled={i === steps.length - 1}
          style={btnStyle(i === steps.length - 1)}
        >
          Next →
        </button>
      </div>
    </div>
  );
}

function btnStyle(disabled) {
  return {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 11,
    padding: "4px 10px",
    background: "transparent",
    color: disabled ? colors.textDark : colors.gold,
    border: `1px solid ${disabled ? colors.border : colors.gold + "55"}`,
    borderRadius: 3,
    cursor: disabled ? "default" : "pointer",
  };
}
