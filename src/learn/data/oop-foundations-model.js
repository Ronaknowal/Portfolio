// Deliberately bounded teaching models, independently checked against Python.
export const receiverObjects = { morning: "A", evening: "B", alias: "A" };

export function initialObjectState() {
  return { A: [], B: [] };
}

export function methodCallFrames(state, receiver, value) {
  if (!(receiver in receiverObjects) || ![18, 24, 30].includes(value)) throw new Error("Unsupported call");
  const object = receiverObjects[receiver];
  const before = { A: [...state.A], B: [...state.B] };
  const after = { ...before, [object]: [...before[object], value] };
  return [
    { state: before, object, title: "Predict before calling", note: `Which list will ${receiver}.add(${value}) change? Follow the name's arrow before stepping.` },
    { state: before, object, title: "1 · Find the receiving object", note: `${receiver} refers to object ${object}. A name identifies an object; it does not contain a separate copy of that object's data.` },
    { state: before, object, title: "2 · Bind the method", note: `The class supplies the add function. The bound method pairs that function with object ${object}; no reading has been added yet.` },
    { state: before, object, title: "3 · Enter the function", note: `Inside this call, self refers to object ${object} and value is ${value}. self is a local parameter, not a third log.` },
    { state: after, object, title: "4 · Mutate the selected list", note: `self.values reaches list ${object}. append adds ${value} there. The other log's list stays unchanged. The method returns None because it has no return statement.` },
  ];
}

export function initialLookupState(mode = "class") {
  return { mode, shared: [], ownA: mode === "instance" ? [] : null, ownB: mode === "instance" ? [] : null };
}

export function lookupValues(state, receiver) {
  if (!["A", "B"].includes(receiver)) throw new Error("Unsupported receiver");
  const own = state[`own${receiver}`];
  return { source: own === null ? "class" : "instance", values: own === null ? state.shared : own, listId: own === null ? "C" : receiver };
}

export function lookupAction(state, receiver, action) {
  if (!["append", "assign"].includes(action)) throw new Error("Unsupported action");
  const found = lookupValues(state, receiver);
  const next = { ...state, shared: [...state.shared], ownA: state.ownA === null ? null : [...state.ownA], ownB: state.ownB === null ? null : [...state.ownB] };
  if (action === "assign") next[`own${receiver}`] = [99];
  else if (found.source === "class") next.shared.push(18);
  else next[`own${receiver}`].push(18);
  return next;
}

export const readingCandidates = [
  { id: "number", code: "24", label: "24 · int", value: 24, type: "int" },
  { id: "bool", code: "True", label: "True · bool", value: true, type: "bool" },
  { id: "text", code: '"24"', label: '"24" · str', value: "24", type: "str" },
  { id: "nan", code: 'float("nan")', label: "NaN · float", value: NaN, type: "float" },
  { id: "infinity", code: 'float("inf")', label: "+infinity · float", value: Infinity, type: "float" },
  { id: "huge", code: "10 ** 400", label: "10 ** 400 · int", type: "int", overflow: true },
];

export function validateReading(id) {
  const candidate = readingCandidates.find(item => item.id === id);
  if (!candidate) throw new Error("Unsupported reading");
  const gates = [
    { label: "Accept int or float, excluding bool", status: "waiting" },
    { label: "Convert to float storage", status: "waiting" },
    { label: "Require a finite value", status: "waiting" },
    { label: "Append after all checks pass", status: "waiting" },
  ];
  let error = null;
  if (!["int", "float"].includes(candidate.type)) { gates[0].status = "blocked"; error = "TypeError: expected int or float, not bool"; }
  else {
    gates[0].status = "passed";
    if (candidate.overflow) { gates[1].status = "blocked"; error = "ValueError: reading does not fit float storage"; }
    else {
      gates[1].status = "passed";
      if (!Number.isFinite(candidate.value)) { gates[2].status = "blocked"; error = "ValueError: reading must be finite"; }
      else { gates[2].status = "passed"; gates[3].status = "passed"; }
    }
  }
  return { candidate, gates, error, before: [18], after: error ? [18] : [18, Number(candidate.value)] };
}

export function formatReading(value, formatter) {
  if (!["celsius", "fahrenheit"].includes(formatter) || ![0, 20, 30].includes(value)) throw new Error("Unsupported formatter input");
  return formatter === "celsius" ? `${value.toFixed(1)} C` : `${(value * 9 / 5 + 32).toFixed(1)} F`;
}

export function compositionFrames(value, formatter) {
  const output = formatReading(value, formatter);
  const name = formatter === "celsius" ? "CelsiusFormatter" : "FahrenheitFormatter";
  return [
    { title: "Predict the report", note: `The input remains ${value} degrees Celsius. Predict the text returned by ${name}.`, output: null },
    { title: "1 · Ask the report", note: `The caller sends ${value} to report.render. Report stores a reference to a formatter, not a copy of its formatting code.`, output: null },
    { title: "2 · Delegate one responsibility", note: `Report calls self.formatter.format(${value}). The receiving object is now the ${name} instance, so self inside format is the formatter.`, output: null },
    { title: "3 · Return text to the report", note: `${name} returns "${output}". A different formatter may compute differently while keeping the same input/output contract.`, output },
    { title: "4 · Return the finished report", note: `Report adds its label and returns "Reading: ${output}". No log state changes and the Celsius input is unchanged.`, output: `Reading: ${output}` },
  ];
}
