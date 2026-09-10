const expressionNodes = [
  { id: "product", label: "×", x: 180, y: 28 },
  { id: "sum", label: "+", x: 90, y: 104 },
  { id: "difference", label: "−", x: 270, y: 104 },
  { id: "two", label: "2", x: 35, y: 186 },
  { id: "three", label: "3", x: 145, y: 186 },
  { id: "nine", label: "9", x: 215, y: 186 },
  { id: "four", label: "4", x: 325, y: 186 },
];
const expressionEdges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]];

export default function ExpressionTreeFigure() {
  return <figure className="tree-inline-figure">
    <svg viewBox="0 0 360 224" role="img" aria-label="Multiplication root with addition of 2 and 3 on the left and subtraction of 9 and 4 on the right" style={{ display: "block", width: "100%", maxWidth: 480, margin: "auto" }}>
      {expressionEdges.map(([from, to]) => <line key={`${from}-${to}`} x1={expressionNodes[from].x} y1={expressionNodes[from].y} x2={expressionNodes[to].x} y2={expressionNodes[to].y} stroke="#afa78b" strokeWidth="2" />)}
      {expressionNodes.map((node, index) => <g key={node.id}>
        <circle cx={node.x} cy={node.y} r="23" fill={index < 3 ? "#29251b" : "#152720"} stroke={index < 3 ? "#eab94e" : "#82b99d"} strokeWidth="2" />
        <text x={node.x} y={node.y + 7} textAnchor="middle" fill="#f3eee1" fontSize="22">{node.label}</text>
      </g>)}
    </svg>
    <figcaption>Leaves supply values. The left branch returns 2 + 3 = 5; the right returns 9 − 4 = 5. Only then can the root return 5 × 5 = 25. These links encode grouping, not sorted-key order.</figcaption>
  </figure>;
}
