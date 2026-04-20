import katex from "katex";
import { colors, fonts } from "../../styles";

export function Math({ children }) {
  const html = katex.renderToString(String(children), {
    displayMode: false,
    throwOnError: false,
  });
  return (
    <span
      style={{ fontSize: 13 }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function MathBlock({ children, caption }) {
  const html = katex.renderToString(String(children), {
    displayMode: true,
    throwOnError: false,
  });
  return (
    <div style={{ margin: "20px 0", textAlign: "center" }}>
      <div dangerouslySetInnerHTML={{ __html: html }} />
      {caption && (
        <div style={{
          fontFamily: fonts.mono,
          fontSize: 11,
          color: colors.textDim,
          marginTop: 8,
        }}>
          {caption}
        </div>
      )}
    </div>
  );
}
