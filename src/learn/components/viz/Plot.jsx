import { colors, fonts } from "../../styles";

// series: Array<{ name: string, color?: string, points: Array<[x, y]> }>
export default function Plot({ series, width = 480, height = 240, xLabel, yLabel, label }) {
  const allPts = series.flatMap((s) => s.points);
  const xs = allPts.map(([x]) => x);
  const ys = allPts.map(([, y]) => y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const pad = { t: 16, r: 16, b: 40, l: 48 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  const sx = (x) => pad.l + ((x - xMin) / xRange) * plotW;
  const sy = (y) => pad.t + plotH - ((y - yMin) / yRange) * plotH;

  const palette = [colors.gold, colors.green, "#c084fc", "#60a5fa"];

  return (
    <div style={{ margin: "16px 0" }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 6, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}
      <svg width={width} height={height} style={{ background: "rgba(0,0,0,0.2)", border: `1px solid ${colors.border}`, borderRadius: 4 }}>
        <line x1={pad.l} y1={pad.t + plotH} x2={pad.l + plotW} y2={pad.t + plotH} stroke={colors.textDark} />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + plotH} stroke={colors.textDark} />

        {xLabel && <text x={pad.l + plotW / 2} y={height - 8} fill={colors.textMuted} fontSize={11} fontFamily={fonts.mono} textAnchor="middle">{xLabel}</text>}
        {yLabel && <text x={12} y={pad.t + plotH / 2} fill={colors.textMuted} fontSize={11} fontFamily={fonts.mono} transform={`rotate(-90 12 ${pad.t + plotH / 2})`} textAnchor="middle">{yLabel}</text>}

        <text x={pad.l} y={pad.t + plotH + 14} fill={colors.textDim} fontSize={10} fontFamily={fonts.mono} textAnchor="middle">{xMin}</text>
        <text x={pad.l + plotW} y={pad.t + plotH + 14} fill={colors.textDim} fontSize={10} fontFamily={fonts.mono} textAnchor="middle">{xMax}</text>
        <text x={pad.l - 6} y={pad.t + plotH + 3} fill={colors.textDim} fontSize={10} fontFamily={fonts.mono} textAnchor="end">{yMin}</text>
        <text x={pad.l - 6} y={pad.t + 3} fill={colors.textDim} fontSize={10} fontFamily={fonts.mono} textAnchor="end">{yMax}</text>

        {series.map((s, i) => {
          const color = s.color || palette[i % palette.length];
          const d = s.points.map(([x, y], idx) => `${idx === 0 ? "M" : "L"} ${sx(x)} ${sy(y)}`).join(" ");
          return (
            <g key={i}>
              <path d={d} stroke={color} fill="none" strokeWidth={1.5} />
              {s.points.map(([x, y], idx) => (
                <circle key={idx} cx={sx(x)} cy={sy(y)} r={2} fill={color} />
              ))}
            </g>
          );
        })}

        {series.length > 1 && (
          <g>
            {series.map((s, i) => {
              const color = s.color || palette[i % palette.length];
              return (
                <g key={i} transform={`translate(${pad.l + 8} ${pad.t + 12 + i * 14})`}>
                  <rect width={10} height={2} y={4} fill={color} />
                  <text x={16} y={8} fill={colors.textSecondary} fontSize={10} fontFamily={fonts.mono}>{s.name}</text>
                </g>
              );
            })}
          </g>
        )}
      </svg>
    </div>
  );
}
