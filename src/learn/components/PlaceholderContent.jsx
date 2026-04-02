import { colors, fonts } from "../styles";

export default function PlaceholderContent({ title }) {
  return (
    <div style={{ padding: "40px 0" }}>
      <div style={{
        border: `1px dashed ${colors.border}`,
        borderRadius: 8,
        padding: "48px 32px",
        textAlign: "center",
      }}>
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDark, letterSpacing: 2, marginBottom: 12 }}>
          WEIGHTS NOT YET INITIALIZED
        </div>
        <div style={{ fontFamily: fonts.sans, fontSize: 18, color: colors.textMuted, marginBottom: 8 }}>
          {title}
        </div>
        <div style={{ fontFamily: fonts.mono, fontSize: 11, color: colors.textDim }}>
          Content coming soon — this topic is queued for training.
        </div>
      </div>
    </div>
  );
}
