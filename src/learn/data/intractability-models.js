// Finite teaching models. No execution time or general lower bound is inferred from enumeration.
export const formulaPresets = {
  satisfiable: [[1, 2, 3], [-1, 2, -3], [1, -2, -3]],
  contradictory: [[1], [-1]],
  repeated: [[1, -1, 2], [2, 2, -3], [-2, 3, 1]]
};
export const coverPresets = {
  trianglePath: [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 5]],
  star: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
  empty: []
};
export function literalName(literal) {
  return `${literal < 0 ? '¬' : ''}x${Math.abs(literal)}`;
}
export function formatFormula(clauses) {
  return clauses.map(clause => clause.join(' ')).join('\n');
}
export function parseFormula(text) {
  if (typeof text !== 'string' || text.length > 300) throw new Error('Use at most 300 characters.');
  const lines = text.trim().split(/\n/).filter(line => line.trim());
  if (lines.length < 1 || lines.length > 3) throw new Error('Enter 1–3 nonempty clause lines.');
  const clauses = lines.map(line => line.trim().split(/\s+/).map(token => {
    if (!/^-?[123]$/.test(token)) throw new Error('Use 1, 2, 3 or their negatives for x1, x2, x3.');
    return Number(token);
  }));
  if (clauses.some(clause => clause.length > 3)) throw new Error('Each clause has 1–3 literal occurrences.');
  return clauses;
}
function checkFormula(clauses) {
  if (!Array.isArray(clauses) || clauses.length > 3 || clauses.some(clause => !Array.isArray(clause) || clause.length > 3 || clause.some(literal => !Number.isInteger(literal) || ![1, 2, 3].includes(Math.abs(literal))))) {
    throw new Error('The model accepts up to 3 clauses with up to 3 signed literals from 1…3.');
  }
}
export function evaluateFormula(clauses, assignment) {
  checkFormula(clauses);
  if (!Array.isArray(assignment) || assignment.length !== 3 || assignment.some(value => typeof value !== 'boolean')) throw new Error('Provide three Boolean values.');
  const rows = clauses.map(clause => {
    const values = clause.map(literal => literal > 0 ? assignment[literal - 1] : !assignment[-literal - 1]);
    return {
      literals: [...clause],
      values,
      satisfied: values.some(Boolean)
    };
  });
  return {
    rows,
    satisfied: rows.every(row => row.satisfied)
  };
}
export function truthTable(clauses) {
  const result = [];
  for (const x1 of [false, true]) for (const x2 of [false, true]) for (const x3 of [false, true]) {
    const assignment = [x1, x2, x3];
    result.push({
      assignment,
      ...evaluateFormula(clauses, assignment)
    });
  }
  return result;
}
export function reduceFormulaToClique(clauses) {
  checkFormula(clauses);
  const vertices = clauses.flatMap((clause, clauseIndex) => clause.map((literal, occurrence) => ({
    id: `${clauseIndex}:${occurrence}`,
    clause: clauseIndex,
    occurrence,
    literal,
    x: 55 + clauseIndex * 112,
    y: 55 + occurrence * 65
  })));
  const edges = [];
  for (let i = 0; i < vertices.length; i += 1) for (let j = i + 1; j < vertices.length; j += 1) {
    if (vertices[i].clause !== vertices[j].clause && vertices[i].literal !== -vertices[j].literal) edges.push([vertices[i].id, vertices[j].id]);
  }
  return {
    vertices,
    edges,
    target: clauses.length
  };
}
export function inspectSelection(clauses, selected) {
  const graph = reduceFormulaToClique(clauses);
  if (!Array.isArray(selected) || selected.length !== clauses.length) throw new Error('One choice slot per clause is required.');
  const vertices = selected.map((id, clause) => {
    if (id === null) return null;
    const vertex = graph.vertices.find(item => item.id === id && item.clause === clause);
    if (!vertex) throw new Error('Choice is not an occurrence in its clause.');
    return vertex;
  });
  const pairs = [];
  for (let i = 0; i < vertices.length; i += 1) for (let j = i + 1; j < vertices.length; j += 1) {
    if (vertices[i] && vertices[j]) pairs.push({
      first: vertices[i],
      second: vertices[j],
      compatible: vertices[i].literal !== -vertices[j].literal
    });
  }
  const complete = vertices.every(Boolean);
  const clique = complete && pairs.every(pair => pair.compatible);
  let assignment = null;
  if (clique) {
    assignment = [false, false, false];
    vertices.forEach(vertex => {
      assignment[Math.abs(vertex.literal) - 1] = vertex.literal > 0;
    });
  }
  return {
    graph,
    vertices,
    pairs,
    complete,
    clique,
    assignment
  };
}
export function firstCliqueChoices(clauses) {
  checkFormula(clauses);
  function visit(clause, selected) {
    if (clause === clauses.length) return inspectSelection(clauses, selected).clique ? selected : null;
    for (let occurrence = 0; occurrence < clauses[clause].length; occurrence += 1) {
      const result = visit(clause + 1, [...selected, `${clause}:${occurrence}`]);
      if (result !== null) return result;
    }
    return null;
  }
  return visit(0, []);
}
export function encodingCounts(exponent) {
  if (!Number.isInteger(exponent) || exponent < 1 || exponent > 40) throw new Error('Choose an integer exponent 1…40.');
  const target = 2n ** BigInt(exponent);
  return {
    binary: `1${'0'.repeat(exponent)}`,
    bits: exponent + 1,
    target: target.toString(),
    slots: (target + 1n).toString()
  };
}
export function formatCoverEdges(edges) {
  return edges.map(([u, v]) => `${String.fromCharCode(65 + u)} ${String.fromCharCode(65 + v)}`).join('\n');
}
export function parseCoverEdges(text) {
  if (typeof text !== 'string' || text.length > 400) throw new Error('Use at most 400 characters.');
  const edges = [];
  const seen = new Set();
  for (const line of text.trim().split('\n').filter(line => line.trim())) {
    const match = /^([A-F])\s+([A-F])$/i.exec(line.trim());
    if (!match) throw new Error('Enter two vertex labels A–F per line, separated by a space.');
    const pair = match.slice(1).map(label => label.toUpperCase().charCodeAt(0) - 65).sort((a, b) => a - b);
    if (pair[0] === pair[1]) throw new Error('This simple-graph model excludes self-loops.');
    const key = pair.join(':');
    if (!seen.has(key)) {
      edges.push(pair);
      seen.add(key);
    }
  }
  if (edges.length > 15) throw new Error('At most 15 distinct edges on A–F.');
  return edges;
}
function checkGraph(edges) {
  if (!Array.isArray(edges) || edges.length > 15 || edges.some(edge => !Array.isArray(edge) || edge.length !== 2 || edge.some(vertex => !Number.isInteger(vertex) || vertex < 0 || vertex > 5) || edge[0] === edge[1])) throw new Error('Use a simple graph on vertices 0…5.');
  if (new Set(edges.map(edge => [...edge].sort().join(':'))).size !== edges.length) throw new Error('Duplicate edge.');
}
export function isCover(edges, selected) {
  const set = new Set(selected);
  return edges.every(([u, v]) => set.has(u) || set.has(v));
}
export function matchingCover(edges) {
  checkGraph(edges);
  const selected = new Set();
  const matching = [];
  const states = [{
    selected: [],
    matching: [],
    considered: null
  }];
  edges.forEach(([u, v], index) => {
    if (!selected.has(u) && !selected.has(v)) {
      selected.add(u);
      selected.add(v);
      matching.push(index);
    }
    states.push({
      selected: [...selected],
      matching: [...matching],
      considered: index
    });
  });
  return {
    selected: [...selected],
    matching,
    states
  };
}
export function exactCover(edges, budget) {
  checkGraph(edges);
  if (!Number.isInteger(budget) || budget < 0 || budget > 6) throw new Error('Budget is an integer 0…6.');
  const events = [];
  function visit(remaining, left, selected, depth) {
    events.push({
      remaining: remaining.map(edge => [...edge]),
      left,
      selected: [...selected],
      depth
    });
    if (remaining.length === 0) return selected;
    if (left === 0) return null;
    const [u, v] = remaining[0];
    for (const chosen of [u, v]) {
      const answer = visit(remaining.filter(edge => !edge.includes(chosen)), left - 1, [...selected, chosen], depth + 1);
      if (answer !== null) return answer;
    }
    return null;
  }
  return {
    selected: visit(edges, budget, [], 0),
    events
  };
}
export function minimumCover(edges) {
  checkGraph(edges);
  for (let budget = 0; budget <= 6; budget += 1) {
    const result = exactCover(edges, budget);
    if (result.selected !== null) return result.selected;
  }
  throw new Error('Every finite simple graph has a cover.');
}
