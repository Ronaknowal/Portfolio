// Small, deterministic teaching models. These do not interpret arbitrary Python.
export function referenceTrace(copy = false, action = "append") {
  const states = [{ names: {}, objects: {}, line: -1, explanation: "No names have been assigned yet. Step once to create a list." }];
  let names = {}, objects = {};
  function capture(line, explanation) {
    states.push({ names: { ...names }, objects: structuredClone(objects), line, explanation });
  }
  objects.A = [18, 21]; names.readings = "A";
  capture(0, "Python creates list A. The name readings refers to it; the name is not the container.");
  if (copy) { objects.B = [...objects.A]; names.backup = "B"; }
  else names.backup = "A";
  capture(1, copy ? "copy() creates a second outer list B. Equal contents do not imply one shared object." : "Assignment adds a second name pointing to list A. No list is copied.");
  if (action === "append") {
    objects[names.backup].push(24);
    capture(2, copy ? "append changes B in place. A stays unchanged because these are separate lists." : "append changes A in place. Both names now see the extra reading because both point to A.");
  } else {
    objects.C = [0]; names.backup = "C";
    capture(2, "Assignment makes backup point to new list C. It does not edit A or move the readings reference. An unreferenced teaching object is hidden below; this diagram does not model garbage-collection timing.");
  }
  capture(3, `readings observes [${objects[names.readings].join(", ")}]; backup observes [${objects[names.backup].join(", ")}].`);
  return { code: ["readings = [18, 21]", copy ? "backup = readings.copy()" : "backup = readings", action === "append" ? "backup.append(24)" : "backup = [0]", "print(readings, backup)"], states };
}

export const loopReadings = [18, null, 25, 31, 0];
export function selectionTrace(threshold = 20, readings = loopReadings) {
  if (!Number.isFinite(threshold) || readings.some(v => v !== null && !Number.isFinite(v))) throw new Error("Use finite numeric readings or null in this teaching model.");
  const states = [], accepted = [], decisions = [];
  const capture = (line, index, phase, explanation) => states.push({ line, index, phase, accepted: [...accepted], decisions: [...decisions], explanation });
  capture(1, -1, "ready", "The result list begins empty. The input stays unchanged throughout the loop.");
  readings.forEach((value, index) => {
    capture(2, index, "take", `The loop binds value to input position ${index}: ${value === null ? "None" : value}.`);
    capture(3, index, "missing", value === null ? "The missing-value test is true. Do not compare None with a temperature." : "This value is present. Zero is present too; only None means missing here.");
    if (value === null) {
      decisions[index] = "missing";
      capture(4, index, "skip", "continue returns control to the loop for the next item. It does not end the whole loop.");
    } else {
      capture(5, index, "compare", `${value} >= ${threshold} is ${value >= threshold ? "True" : "False"}. ${value >= threshold ? "Follow the add branch." : "Skip the append and return to the loop."}`);
      decisions[index] = value >= threshold ? "kept" : "below";
      if (value >= threshold) {
        accepted.push(value);
        capture(6, index, "append", `Append ${value} to selected. Only accepted readings reach this container.`);
      } else capture(2, index, "reject", `${value} stays in the input but is not added to selected.`);
    }
  });
  capture(7, readings.length, "done", `The input is exhausted. The loop ends and print displays [${accepted.join(", ")}].`);
  return { code: [`readings = [${readings.map(v => v === null ? "None" : v).join(", ")}]`, "selected = []", "for value in readings:", "    if value is None:", "        continue", `    if value >= ${threshold}:`, "        selected.append(value)", "print(selected)"], states };
}

export function functionTrace(celsius = 20, mode = "return") {
  const converted = celsius * 9 / 5 + 32;
  const returned = mode === "return" ? converted : null;
  const code = ["def convert(celsius):", "    fahrenheit = celsius * 9 / 5 + 32", mode === "return" ? "    return fahrenheit" : "    print(fahrenheit)", `result = convert(${celsius})`, "print(result)"];
  return { code, states: [
    { line: 0, stage: "defined", frame: false, output: [], explanation: "def creates the function. Its body has not run, so celsius and fahrenheit do not exist in a call yet." },
    { line: 3, stage: "call", frame: true, celsius, output: [], explanation: `The caller pauses the assignment. A new call frame binds the parameter celsius to argument ${celsius}. result has not been assigned yet.` },
    { line: 1, stage: "calculate", frame: true, celsius, fahrenheit: converted, output: [], explanation: `The function computes ${celsius} × 9 / 5 + 32 = ${converted}. fahrenheit is a local name in this invocation.` },
    { line: 2, stage: mode, frame: mode !== "return", celsius, fahrenheit: converted, returned: mode === "return" ? converted : undefined, output: mode === "print" ? [converted] : [], explanation: mode === "return" ? `return ends this invocation and sends ${converted} to the waiting caller. It has displayed nothing.` : `print displays ${converted}. Displaying it does not give that number to the caller; print itself returns None.` },
    { line: 3, stage: "assigned", frame: false, result: returned, returned, output: mode === "print" ? [converted] : [], explanation: mode === "return" ? `The call expression evaluates to ${converted}. The caller now binds result to that value.` : "The function reaches its end without return, so the call evaluates to None. The caller binds result to None." },
    { line: 4, stage: "done", frame: false, result: returned, returned, output: mode === "print" ? [converted, null] : [converted], explanation: mode === "return" ? "The caller prints its result. The number was available for another calculation before being displayed." : "The caller prints None. The earlier displayed number cannot be retrieved by assigning the call's result." },
  ] };
}
