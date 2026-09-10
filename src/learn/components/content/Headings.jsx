import { colors, fonts } from "../../styles";

const teachingLabels = {
  "1. Why it exists": "1. Start with the problem",
  "2. Core intuition": "2. Build the intuition",
  "3. Math foundation": "3. Make the idea precise",
  "3. Mathematical foundation": "3. Make the idea precise",
  "4. How it works": "4. See how it works",
  "5. Implementation": "5. Put it into practice",
  "6. Failure modes": "6. Know where it can fail",
};

export function H2({ children }) {
  const label = typeof children === "string" ? (teachingLabels[children] || children) : children;
  const id = typeof label === "string"
    ? label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")
    : undefined;
  return (
    <h2 id={id} style={{
      fontFamily: fonts.sans,
      fontSize: 22,
      fontWeight: 600,
      color: colors.textPrimary,
      margin: "32px 0 14px",
    }}>
      {label}
    </h2>
  );
}

export function H3({ children }) {
  return (
    <h3 style={{
      fontFamily: fonts.sans,
      fontSize: 20,
      fontWeight: 600,
      color: colors.textPrimary,
      margin: "24px 0 12px",
    }}>
      {children}
    </h3>
  );
}
