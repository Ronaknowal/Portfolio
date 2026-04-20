import { colors, fonts } from "../../styles";

// text: string
export default function ByteGrid({ text, label }) {
  const bytes = Array.from(new TextEncoder().encode(text));

  const assoc = [];
  for (const ch of text) {
    const n = new TextEncoder().encode(ch).length;
    for (let k = 0; k < n; k++) assoc.push({ ch, byteIndex: k, byteCount: n });
  }

  return (
    <div style={{ margin: "16px 0" }}>
      {label && (
        <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginBottom: 6, letterSpacing: 1 }}>
          {label.toUpperCase()}
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
        {bytes.map((b, i) => {
          const a = assoc[i] || { ch: "?" };
          const multi = a.byteCount > 1;
          return (
            <div
              key={i}
              title={`Byte ${i}: 0x${b.toString(16).padStart(2, "0")}  ·  char "${a.ch}" (${a.byteCount} byte${a.byteCount > 1 ? "s" : ""})`}
              style={{
                fontFamily: fonts.mono,
                fontSize: 10,
                padding: "4px 6px",
                borderRadius: 2,
                background: multi ? "rgba(226,181,90,0.12)" : "rgba(74,222,128,0.08)",
                border: `1px solid ${multi ? colors.gold + "55" : colors.green + "55"}`,
                color: multi ? colors.gold : colors.green,
                minWidth: 28,
                textAlign: "center",
              }}
            >
              {b.toString(16).padStart(2, "0")}
            </div>
          );
        })}
      </div>
      <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.textDim, marginTop: 6 }}>
        <span style={{ color: colors.green }}>green</span> = ASCII (1 byte) ·
        <span style={{ color: colors.gold }}> gold</span> = multi-byte UTF-8
      </div>
    </div>
  );
}
