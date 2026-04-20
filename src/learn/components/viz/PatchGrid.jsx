import { colors, fonts } from "../../styles";

// src: image URL
// patches: integer — number of patches per side
export default function PatchGrid({ src, patches = 8, size = 240, label }) {
  const cells = [];
  for (let r = 0; r < patches; r++) {
    for (let c = 0; c < patches; c++) cells.push([r, c]);
  }

  return (
    <div style={{ margin: "16px 0" }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 6, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}
      <div style={{ position: "relative", width: size, height: size, border: `1px solid ${colors.border}` }}>
        {src && (
          <img
            src={src}
            alt=""
            style={{ width: size, height: size, objectFit: "cover", display: "block", opacity: 0.85 }}
          />
        )}
        <svg width={size} height={size} style={{ position: "absolute", inset: 0 }}>
          {cells.map(([r, c]) => (
            <rect
              key={`${r}-${c}`}
              x={(c * size) / patches}
              y={(r * size) / patches}
              width={size / patches}
              height={size / patches}
              fill="none"
              stroke={colors.gold + "66"}
              strokeWidth={0.5}
            />
          ))}
        </svg>
      </div>
      <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginTop: 6 }}>
        {patches}×{patches} = {patches * patches} patches
      </div>
    </div>
  );
}
