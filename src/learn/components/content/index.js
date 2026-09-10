export { Prose } from "./Prose";
export { H2, H3 } from "./Headings";
export { Code, CodeBlock } from "./Code";
export { Callout } from "./Callout";
export { Figure } from "./Figure";
// Keep ordinary prose/code imports independent of the math renderer.
// Formula consumers import Math/MathBlock directly from ./Math.jsx.
