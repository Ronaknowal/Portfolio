import { colors, fonts } from "../../styles";

export function Figure({ children, caption }) {
  return (
    <figure style={{ margin: "24px 0" }}>
      {children}
      {caption && (
        <figcaption style={{
          fontFamily: fonts.mono,
          fontSize: 11,
          color: colors.textDim,
          marginTop: 8,
          textAlign: "center",
        }}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
