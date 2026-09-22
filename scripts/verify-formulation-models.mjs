// Bounded independent checks of the problem-formulation browser models against
// the content packet's recorded calculations, the manuscript's stated values,
// analytic identities, and a second derivation for every claim a figure draws.
//
// No loop here is allowed to inspect an empty subject set, and every group
// asserts how many subjects it actually saw.
//
// What this file does NOT claim is that nothing is compared with itself. It
// used to. Several dozen geometry assertions compare a drawn coordinate
// against the scale that produced it -- `scale(clamp(v))` against the endpoints
// of that same scale's range. They still catch a source mutation, which is what
// the falsification harness exercises and what case 3 fires on, and the real
// containment check is the browser verifier's against the rendered SVG box. But
// they are not independent, and a blanket claim that they were is the kind of
// sentence this effort keeps having to retract.
//
// Three rules shaped this file.
//
//   1. A SECOND ROUTE MEANS A DIFFERENT DERIVATION, AND THIS FILE OWNS ONE END
//      OF IT. Average precision, log loss and the confusion cells are
//      implemented locally below -- `averagePrecisionLocal`, `logLossLocal`,
//      `confusionLocal` -- and each measured value is checked three ways: the
//      module under test, this file's own implementation, and scikit-learn's
//      recorded output in the frozen packet. The ranking has two local routes,
//      an explicit sort and a quadratic repeated-maximum. The as-known
//      selection is checked against a route that sorts a zero-padded composite
//      key and takes the last element rather than folding a comparison, and
//      against the manuscript's own program output. The constant baseline's
//      average precision is checked against its closed form, the prevalence.
//
//      The header used to describe those metric implementations as living here
//      when all four were imported from the module under test and called. The
//      comparisons were sound -- the other side was scikit-learn's -- but a
//      reader auditing for an in-repo second route would have found a claim
//      rather than a route. There is a route now.
//
//   2. A RULE DRAWN MUST EQUAL THE RULE APPLIED ACROSS THE WHOLE ENTERABLE
//      GRID. The as-known selection is swept over every cutoff, every maximum
//      age, both entities and a wide family of arrival configurations --
//      50,000+ cases -- and the capacity rule over every capacity the control
//      offers for all three rankings, plus tens of thousands of exchanges.
//      A lab that grades a correct answer wrong is the defect class this topic
//      is most exposed to, and on this topic it would also be the lesson's own
//      subject matter.
//
//   3. EVERY GRADED COMPARISON IS EXERCISED AT ITS DEGENERATE INPUTS. A cutoff
//      of 0, an age limit of 0, an arrival exactly at the cutoff, an event
//      exactly at the age boundary, a capacity of 1, a capacity equal to the
//      evaluated set, an exchange that changes nothing, and a difference of
//      exactly the tolerance all appear below, and the unchanged rule is
//      asserted as an IF AND ONLY IF rather than by example.
//
// Run: node scripts/verify-formulation-models.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  AVAILABILITY_FEATURES, LABEL_ASCENT, availabilityAxisGeometry, averagePrecision, checkFinite, confusionAtThreshold,
  costThreshold, criticalQuantile, expectedLosses, flowGeometry, flowStages, lineageGeometry, lineagePaths,
  latestKnown, linearScale, logLoss, majorityBaseline, membershipChange, metricAxes, metricBarGeometry,
  displayTolerance, gradingAllowance,
  movementOf, nestedPartitionGeometry, optimalStock, parcelFixture, partitionGeometry, policyOrder,
  policyOutcome, practiceTimelineFixture, rankByScore, recordEligibility, recordKey, recordLabel,
  reverseSelected, selectedStripGeometry, selectionMetrics, splitLaneGeometry, stockingCost, supportFixture,
  swapSelection, thresholdPlotGeometry, timelineFixture, timelineGeometry, timelineLabelCollisions,
  translateValues, unchangedTolerance, upliftGeometry, upliftGroups, validateRecord, withArrival, withValue,
} from '../src/learn/data/formulation-models.js';
import { formulationData } from '../src/learn/data/formulation-data.js';
import { formulationExamples } from '../src/learn/data/formulation-examples.js';

const packetDirectory = 'docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage';
const recorded = JSON.parse(fs.readFileSync(`${packetDirectory}/calculated-inputs.json`, 'utf8'));
const manuscript = fs.readFileSync(`${packetDirectory}/lesson.md`, 'utf8');
const lessonBody = fs.readFileSync('src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx', 'utf8');
const labsBodyForGuards = fs.readFileSync('src/learn/components/lesson-labs/FormulationLabs.jsx', 'utf8');

const evidencePath = 'docs/teaching/evidence/formulation-models.json';
const keepEvidence = !process.argv.includes('--no-evidence');
/* A PROVISIONAL record, written before the first assertion runs.
 *
 * Writing evidence only at the end looks safe and is not: a run that fails
 * leaves the PREVIOUS file on disk, still saying `passed: true`, describing a
 * source version that no longer exists. Anyone reading the directory then sees
 * a green record for a red tree. The file is stamped `passed: false` here and
 * only becomes true after the last assertion has run. */
if (keepEvidence) {
  fs.mkdirSync('docs/teaching/evidence', { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify({
    startedAt: new Date().toISOString(),
    verifier: 'scripts/verify-formulation-models.mjs',
    status: 'running',
    note: 'Provisional record written before the first assertion. If this is what is on disk, the run did not '
      + 'reach its end: it threw, or it was killed.',
    passed: false,
  }, null, 2) + '\n');
}

const counts = {};
const record = name => { counts[name] = (counts[name] ?? 0) + 1; };
const close = (actual, expected, label, tolerance = 1e-12) => {
  assert(typeof actual === 'number' && Number.isFinite(actual), `${label}: ${actual} is not a finite number`);
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} versus ${expected}`);
};
/** A loop must have something to look at, or it asserts nothing. */
const nonEmpty = (collection, expected, label) => {
  assert(collection.length > 0, `${label}: the subject set is empty, so nothing was checked`);
  if (expected !== undefined) {
    assert.equal(collection.length, expected, `${label}: saw ${collection.length}, expected ${expected}`);
  }
};
/** A call that must be refused. A silent default here would be a wrong answer. */
const refuses = (call, label) => {
  assert.throws(call, RangeError, `${label}: should have been refused`);
  record('refused input');
};

/* ======================================= second routes, written from scratch */

/**
 * The as-known selection by a completely different mechanism: filter with three
 * explicit predicates, build a zero-padded composite sort key per surviving
 * record, sort the strings, and take the last. No comparison fold, no reduce,
 * no shared helper with the module under test.
 */
function asKnownBySortedKey(records, entity, cutoff, maximumAge) {
  const pad = value => String(value + 100).padStart(4, '0');
  const survivors = [];
  for (const item of records) {
    if (item.entity !== entity) continue;
    if (item.event > cutoff) continue;
    if (item.event < cutoff - maximumAge) continue;
    if (item.available > cutoff) continue;
    survivors.push({ key: `${pad(item.event)}|${pad(item.available)}|${pad(item.version)}`, item });
  }
  if (survivors.length === 0) return null;
  survivors.sort((left, right) => (left.key < right.key ? -1 : left.key > right.key ? 1 : 0));
  return survivors[survivors.length - 1].item;
}

/** The rank of every identity by a different mechanism: repeatedly take the
 *  maximum of the remaining pool. Quadratic on purpose -- it shares nothing
 *  with a comparison sort. */
function rankBySelection(ids, scoreById) {
  const pool = [...ids];
  const order = [];
  while (pool.length) {
    let bestIndex = 0;
    for (let index = 1; index < pool.length; index += 1) {
      const challenger = pool[index];
      const holder = pool[bestIndex];
      if (scoreById[challenger] > scoreById[holder]
        || (scoreById[challenger] === scoreById[holder] && challenger < holder)) bestIndex = index;
    }
    order.push(pool[bestIndex]);
    pool.splice(bestIndex, 1);
  }
  return order;
}

/** Log loss for a CONSTANT probability, in closed form. */
function constantLogLoss(positives, total, probability) {
  return -(positives * Math.log(probability) + (total - positives) * Math.log(1 - probability)) / total;
}

/* -------------------------------------------- the metrics, implemented here
 *
 * These three were imported from the module under test and called, while the
 * header claimed they were implemented locally. They are now what the header
 * says: written from their definitions, sharing nothing with
 * formulation-models.js, so each measured value has an in-repo second route as
 * well as scikit-learn's recorded output. */

/** Average precision: sum over DISTINCT score thresholds of the recall
 *  increment times the precision there. Accumulated by walking a descending
 *  index order and grouping ties, which is the definition rather than a copy
 *  of the module's loop. */
function averagePrecisionLocal(targets, scores) {
  const positives = targets.reduce((sum, value) => sum + value, 0);
  assert.ok(positives > 0, 'average precision needs a positive case');
  const order = targets.map((_unused, index) => index)
    .sort((left, right) => (scores[right] - scores[left]) || (left - right));
  const groups = [];
  for (const index of order) {
    const last = groups[groups.length - 1];
    if (last && scores[last.at] === scores[index]) last.members.push(index);
    else groups.push({ at: index, members: [index] });
  }
  let hits = 0;
  let taken = 0;
  let previousRecall = 0;
  let total = 0;
  for (const group of groups) {
    for (const index of group.members) { hits += targets[index]; taken += 1; }
    const recall = hits / positives;
    total += (recall - previousRecall) * (hits / taken);
    previousRecall = recall;
  }
  return total;
}

/** Mean negative log likelihood, accumulated over the positives and the
 *  negatives separately so the summation order differs from the module's. */
function logLossLocal(targets, probabilities) {
  let positiveTerm = 0;
  let negativeTerm = 0;
  targets.forEach((target, index) => {
    if (target === 1) positiveTerm -= Math.log(probabilities[index]);
    else negativeTerm -= Math.log(1 - probabilities[index]);
  });
  return (positiveTerm + negativeTerm) / targets.length;
}

/** The four confusion cells, counted by filtering rather than by a switch. */
function confusionLocal(targets, probabilities, threshold) {
  const predicted = probabilities.map(value => (value >= threshold ? 1 : 0));
  const count = (actual, guess) =>
    targets.filter((target, index) => target === actual && predicted[index] === guess).length;
  return [[count(0, 0), count(0, 1)], [count(1, 0), count(1, 1)]];
}

/* ============================================ §3 · the three clocks, swept */

const baseRecords = timelineFixture.records;
assert.equal(baseRecords.length, 4, 'the calibration history has four records');
assert.deepEqual(baseRecords, formulationData.timeline.records,
  'the fixture the labs use is the history the packet recorded');
assert.deepEqual(baseRecords, recorded.timelineInput, 'and the packet record itself');
record('the calibration fixture is the recorded one');

/* Every record carries a distinct identity, which is what makes "which version"
   a question with an answer. */
const keys = baseRecords.map(recordKey);
assert.equal(new Set(keys).size, keys.length, 'the four records have four distinct identities');
assert.ok(keys.some(key => key.endsWith('v2')), 'and one of them is a second version of an earlier event');
record('record identity');

/* Arrival configurations: each record's arrival over its whole legal range with
   the others at their defaults, plus the full product over the two sensor-A
   records the manuscript's story turns on. */
const arrivalConfigurations = [];
baseRecords.forEach((item, index) => {
  for (let available = item.event; available <= 12; available += 1) {
    arrivalConfigurations.push(baseRecords.map((entry, position) =>
      (position === index ? { ...entry, available } : entry)));
  }
});
for (let firstArrival = 1; firstArrival <= 12; firstArrival += 1) {
  for (let secondArrival = 4; secondArrival <= 12; secondArrival += 1) {
    arrivalConfigurations.push(baseRecords.map(entry => {
      if (entry.entity === 'sensor_A' && entry.event === 1 && entry.version === 1) {
        return { ...entry, available: firstArrival };
      }
      if (entry.entity === 'sensor_A' && entry.event === 4) return { ...entry, available: secondArrival };
      return entry;
    }));
  }
}
nonEmpty(arrivalConfigurations, 150, 'arrival configurations swept');

let timelineCases = 0;
let timelineSelected = 0;
let timelineEmpty = 0;
for (const records of arrivalConfigurations) {
  for (let cutoff = 0; cutoff <= 12; cutoff += 1) {
    for (let maximumAge = 0; maximumAge <= 12; maximumAge += 1) {
      for (const entity of ['sensor_A', 'sensor_B']) {
        const mine = latestKnown({ records, entity, cutoff, maximumAge });
        const theirs = asKnownBySortedKey(records, entity, cutoff, maximumAge);
        timelineCases += 1;
        if (mine.selected === null) {
          assert.equal(theirs, null,
            `the two routes disagree at cutoff ${cutoff}, age ${maximumAge}, ${entity}: `
            + `the fold found nothing and the sorted key found ${theirs && recordKey(theirs)}`);
          timelineEmpty += 1;
          continue;
        }
        assert.ok(theirs !== null, `the sorted-key route found nothing where the fold selected `
          + `${mine.selectedKey} at cutoff ${cutoff}, age ${maximumAge}`);
        assert.equal(mine.selectedKey, recordKey(theirs),
          `the two routes select different records at cutoff ${cutoff}, age ${maximumAge}, ${entity}`);
        timelineSelected += 1;
        /* The leakage property itself, asserted over the whole grid rather than
           at a fixture: the selected value never arrives after the cutoff, and
           never measures an event after it. */
        assert.ok(mine.selected.record.available <= cutoff,
          `the selection at cutoff ${cutoff} arrives at ${mine.selected.record.available}, which is later`);
        assert.ok(mine.selected.record.event <= cutoff,
          `the selection at cutoff ${cutoff} measures an event at ${mine.selected.record.event}`);
        assert.ok(mine.selected.record.event >= cutoff - maximumAge,
          `the selection is older than the ${maximumAge}-unit age limit`);
        assert.equal(mine.selected.record.entity, entity, 'and belongs to the entity that was asked about');
      }
    }
  }
}
assert.ok(timelineCases >= 50000, `only ${timelineCases} as-known cases were swept`);
assert.ok(timelineSelected > 1000 && timelineEmpty > 1000,
  `the sweep saw ${timelineSelected} selections and ${timelineEmpty} empty results; both outcomes must occur `
  + 'or half the rule was never exercised');
record('as-known selection by two routes over the whole enterable grid');

/* The six recorded cases, against the packet and against the manuscript. */
const recordedCases = Object.fromEntries(recorded.timelineResults.map(entry => [entry.name, entry.selected]));
const caseInputs = {
  default: { records: baseRecords, cutoff: 5, maximumAge: 5 },
  arrive_earlier: {
    records: withArrival(timelineFixture, { event: 4, version: 1, available: 4 }).records, cutoff: 5, maximumAge: 5,
  },
  latest_known_revision: { records: baseRecords, cutoff: 7, maximumAge: 7 },
  new_event_now_known: { records: baseRecords, cutoff: 9, maximumAge: 9 },
  too_old: { records: baseRecords, cutoff: 5, maximumAge: 2 },
  unrelated_entity_null: {
    records: withValue(timelineFixture, { entity: 'sensor_B', event: 4, version: 1, value: 88 }).records,
    cutoff: 5, maximumAge: 5,
  },
};
nonEmpty(Object.keys(caseInputs), 6, 'the recorded timeline cases');
for (const [name, inputs] of Object.entries(caseInputs)) {
  const mine = latestKnown({ ...inputs, entity: 'sensor_A' });
  if (recordedCases[name] === null) {
    assert.equal(mine.selected, null, `${name}: the packet records no eligible record and the module found one`);
  } else {
    assert.deepEqual(mine.selected.record, recordedCases[name], `${name}: the selected record differs from the packet`);
  }
  record('recorded timeline case');
}
assert.equal(latestKnown({ ...caseInputs.default, entity: 'sensor_A' }).value, 10,
  'the destination note\'s case: at cutoff 5 the admissible value is 10');
assert.equal(latestKnown({ ...caseInputs.arrive_earlier, entity: 'sensor_A' }).value, 20,
  'and moving the arrival from 8 to 4 changes it to 20');
assert.equal(
  latestKnown({ ...caseInputs.default, entity: 'sensor_A' }).selected.record.event, 1,
  'the selected EVENT at cutoff 5 is event 1');
assert.equal(
  latestKnown({ ...caseInputs.arrive_earlier, entity: 'sensor_A' }).selected.record.event, 4,
  'and after the arrival moves it is event 4 — no event time changed, the admissible match did');
assert.equal(latestKnown({ ...caseInputs.latest_known_revision, entity: 'sensor_A' }).selected.record.version, 2,
  'at cutoff 7 the later VERSION of event 1 is the one that had arrived');
assert.equal(latestKnown({ ...caseInputs.too_old, entity: 'sensor_A' }).selected, null,
  'an age limit of 2 leaves nothing eligible rather than falling back to a future value');
record('the destination note\'s worked case');

/* ------- the note's claim as a PROPERTY, not as two pinned points -------
 *
 * The two assertions above pin (cutoff 5, age 5) before and after the arrival
 * moves. The 50,700-case grid proves the two selection ROUTES agree; neither
 * asserts the thing the destination note is about — that moving an arrival,
 * with no event time changed, can change which record is admissible. Swept
 * over every cutoff and age, for both the 8→4 move and its reverse. */
let arrivalMoveCases = 0;
let arrivalMoveChanges = 0;
const movedFixture = withArrival(timelineFixture, { entity: 'sensor_A', event: 4, version: 1, available: 4 });
for (let cutoff = 0; cutoff <= 12; cutoff += 1) {
  for (let maximumAge = 0; maximumAge <= 12; maximumAge += 1) {
    const before = latestKnown({ records: baseRecords, entity: 'sensor_A', cutoff, maximumAge });
    const after = latestKnown({ records: movedFixture.records, entity: 'sensor_A', cutoff, maximumAge });
    // No event time moved, in either history.
    assert.deepEqual(baseRecords.map(record => record.event), movedFixture.records.map(record => record.event),
      'the moved history has the same event times as the original');
    if (before.selectedKey !== after.selectedKey) {
      arrivalMoveChanges += 1;
      // The only record whose arrival moved is A·event4·v1, so it must be the
      // one that became selectable, and it must now be the newer event.
      assert.equal(after.selected.record.event, 4,
        `at cutoff ${cutoff}, age ${maximumAge}, the selection changed to an event other than the one whose `
        + 'arrival moved');
      assert.ok(before.selected === null || after.selected.record.event > before.selected.record.event,
        'and it is a LATER event than the one an earlier arrival displaced');
    }
    /* Earlier availability can only ever add to what is knowable: whatever was
       eligible before is still eligible after. */
    const eligibleBefore = new Set(before.eligible.map(entry => entry.key));
    for (const key of eligibleBefore) {
      assert.ok(after.eligible.some(entry => entry.key === key),
        `moving an arrival EARLIER removed ${key} from the eligible set at cutoff ${cutoff}`);
    }
    arrivalMoveCases += 1;
  }
}
assert.equal(arrivalMoveCases, 169, `${arrivalMoveCases} arrival-move cases ran, expected 169`);
assert.ok(arrivalMoveChanges >= 3,
  `the arrival move changed the selection in only ${arrivalMoveChanges} of ${arrivalMoveCases} cases; `
  + 'if that reaches zero the note\'s whole claim has stopped being demonstrated');
record('moving an arrival changes the admissible match, as a property over the grid');

/* A record whose arrival precedes its event is refused rather than silently
   ordered. */
refuses(() => validateRecord({ entity: 'sensor_A', event: 5, available: 3, version: 1, value: 1 }),
  'an arrival before its own event');
refuses(() => latestKnown({ records: [], entity: 'sensor_A', cutoff: 5, maximumAge: 5 }),
  'an empty history');
refuses(() => recordEligibility(baseRecords[0], { entity: 'sensor_A', cutoff: 5, maximumAge: -1 }),
  'a negative age limit');
refuses(() => recordEligibility(baseRecords[0], { entity: 'sensor_A', cutoff: 5.5, maximumAge: 5 }),
  'a fractional cutoff');

/* Boundary inclusivity, stated as equalities rather than demonstrated near the
   boundary: a record arriving EXACTLY at the cutoff is admissible, and an event
   exactly at the age boundary is within it. */
const boundaryRecord = { entity: 'sensor_A', event: 4, available: 5, version: 1, value: 7 };
assert.equal(recordEligibility(boundaryRecord, { entity: 'sensor_A', cutoff: 5, maximumAge: 1 }).eligible, true,
  'an arrival exactly at the cutoff is admissible, and an event exactly at the age boundary is within it');
assert.equal(recordEligibility(boundaryRecord, { entity: 'sensor_A', cutoff: 5, maximumAge: 0 }).eligible, false,
  'while an age limit of 0 at cutoff 5 excludes an event at 4');
assert.equal(recordEligibility({ ...boundaryRecord, available: 6 }, { entity: 'sensor_A', cutoff: 5, maximumAge: 1 }).eligible,
  false, 'and an arrival one unit after the cutoff is not');
record('boundary inclusivity');

/* The reasons a rejection gives are the reasons the reveal prints. */
const rejected = recordEligibility(baseRecords[1], { entity: 'sensor_A', cutoff: 5, maximumAge: 5 });
assert.equal(rejected.eligible, false);
assert.equal(rejected.reasons.length, 1, 'the late-arriving record is rejected for exactly one reason');
assert.match(rejected.reasons[0], /arrives at 8, after the cutoff 5/,
  'and that reason names the arrival and the cutoff');
const wrongEntity = recordEligibility(baseRecords[3], { entity: 'sensor_A', cutoff: 5, maximumAge: 5 });
assert.match(wrongEntity.reasons[0], /belongs to sensor_B/, 'the other entity is rejected by name');
record('rejection reasons');

/* Two nulls, asserted as properties over the whole grid rather than at one
   fixture: the other entity's VALUE never moves this entity's selection, and a
   common translation of every value moves the selected VALUE by exactly that
   amount while leaving the selected IDENTITY alone. */
let nullCases = 0;
for (let cutoff = 0; cutoff <= 12; cutoff += 1) {
  for (let maximumAge = 0; maximumAge <= 12; maximumAge += 1) {
    const plain = latestKnown({ records: baseRecords, entity: 'sensor_A', cutoff, maximumAge });
    for (const otherValue of [-100, -1, 0, 1, 42, 88, 100]) {
      const edited = withValue(timelineFixture, { entity: 'sensor_B', event: 4, version: 1, value: otherValue });
      const after = latestKnown({ records: edited.records, entity: 'sensor_A', cutoff, maximumAge });
      assert.equal(after.selectedKey, plain.selectedKey,
        `editing sensor B moved sensor A's selection at cutoff ${cutoff}, age ${maximumAge}`);
      assert.equal(after.value, plain.value, 'and its value');
      nullCases += 1;
    }
    for (const shift of [-20, -1, 0, 1, 5, 30]) {
      const shifted = latestKnown({
        records: translateValues(timelineFixture, shift).records, entity: 'sensor_A', cutoff, maximumAge,
      });
      assert.equal(shifted.selectedKey, plain.selectedKey,
        `translating every value by ${shift} moved the selected identity at cutoff ${cutoff}`);
      if (plain.selected === null) {
        assert.equal(shifted.value, null, 'a missing calibration stays missing under a translation');
      } else {
        assert.equal(shifted.value, plain.value + shift,
          'and the selected value moves by exactly the translation, which is a different claim');
      }
      nullCases += 1;
    }
  }
}
assert.ok(nullCases >= 2000, `only ${nullCases} null cases ran`);
record('the two nulls as properties over the grid');

/* Widening the cutoff with no effective age limit can only ADD eligible
   records: knowledge accumulates. */
let monotoneCases = 0;
for (const records of arrivalConfigurations.slice(0, 40)) {
  let previous = new Set();
  for (let cutoff = 0; cutoff <= 12; cutoff += 1) {
    const eligible = new Set(latestKnown({ records, entity: 'sensor_A', cutoff, maximumAge: 12 })
      .eligible.map(entry => entry.key));
    for (const key of previous) {
      assert.ok(eligible.has(key), `record ${key} became ineligible when the cutoff rose to ${cutoff}`);
    }
    previous = eligible;
    monotoneCases += 1;
  }
}
assert.ok(monotoneCases >= 500, `only ${monotoneCases} monotonicity cases ran`);
record('eligibility is monotone in the cutoff when no age limit binds');

/* The practice fixture, which has its own answers. */
const practice = latestKnown(practiceTimelineFixture);
assert.equal(practice.value, 9, 'practice 1 at cutoff 6 with age 5 selects the value 9');
assert.equal(practice.selected.record.version, 2, 'which is the revision of event 2, not its original');
const practiceEarlier = latestKnown(withArrival(practiceTimelineFixture, { event: 4, version: 1, available: 4 }));
assert.equal(practiceEarlier.value, 11, 'moving the event-4 arrival to 4 makes 11 eligible and newest');
assert.equal(latestKnown({ ...practiceTimelineFixture, maximumAge: 1 }).selected, null,
  'an age limit of 1 leaves nothing');
assert.equal(latestKnown({
  ...withArrival(practiceTimelineFixture, { event: 4, version: 1, available: 4 }), maximumAge: 1,
}).selected, null, 'in either arrival case, which is what makes it a null rather than a second answer');
assert.notEqual(practice.value, practiceEarlier.value,
  'and the two arrival cases really do differ, so the practice question has content');
record('practice 1 fixture');

/* S4: the preset must move the record it NAMES, whatever entity is queried.
   `withArrival` matched on the QUERIED entity, so with the selector on sensor B
   the button named for sensor A's delayed record silently moved sensor B's --
   whose arrival was already 4 -- and left A's at 8. The control carrying this
   section's whole point was a no-op one click away. */
for (const queried of ['sensor_A', 'sensor_B']) {
  const moved = withArrival({ ...timelineFixture, entity: queried },
    { entity: 'sensor_A', event: 4, version: 1, available: 4 });
  const target = moved.records.find(record => record.entity === 'sensor_A' && record.event === 4);
  assert.equal(target.available, 4,
    `the arrive-earlier preset left sensor A's delayed record at ${target.available} while the selector was `
    + `on ${queried}`);
  const untouched = moved.records.find(record => record.entity === 'sensor_B');
  assert.equal(untouched.available, timelineFixture.records.find(r => r.entity === 'sensor_B').available,
    'and moved nothing belonging to the other entity');
  record('the arrive-earlier preset acts on the record it names');
}
refuses(() => withArrival(timelineFixture, { entity: 'sensor_C', event: 4, version: 1, available: 4 }),
  'moving a record that does not exist');

/* S2: the truncation flag is true exactly when the window reaches back past the
   ruler, and the drawn extent is the clamped one. It was computed and read by
   nothing, so a caption could say "-10 to 2" beside a band spanning 0 to 2. */
let truncationCases = 0;
let truncationsSeen = 0;
for (let cutoff = 0; cutoff <= 12; cutoff += 1) {
  for (let maximumAge = 0; maximumAge <= 12; maximumAge += 1) {
    const geometry = timelineGeometry({ ...timelineFixture, cutoff, maximumAge });
    const window = geometry.ageWindow;
    assert.equal(window.startsBeforeTheRuler, window.from < geometry.domain[0],
      `the truncation flag disagrees with the window at cutoff ${cutoff}, age ${maximumAge}`);
    assert.equal(window.drawnFrom, Math.min(Math.max(window.from, geometry.domain[0]), geometry.domain[1]),
      'and the drawn start is the clamped one');
    if (window.startsBeforeTheRuler) {
      assert.ok(window.drawnFrom > window.from, 'a truncated window really is drawn shorter than it is');
      truncationsSeen += 1;
    }
    truncationCases += 1;
  }
}
assert.equal(truncationCases, 169, `${truncationCases} truncation cases ran, expected 169`);
assert.ok(truncationsSeen > 20,
  `only ${truncationsSeen} of ${truncationCases} windows reach past the ruler; if that reaches zero the flag `
  + 'is unreachable and should be deleted rather than reported');
/* And the component has to READ it. A model value nothing consumes is a claim
   nobody checks -- which is what this was. */
assert.ok(labsBodyForGuards.includes('startsBeforeTheRuler'),
  'the timeline figure reads the truncation flag rather than leaving it computed and unused');
record('the age-window truncation flag is correct AND consumed');

/* The displayed program's own printed output. */
const lookupOutput = formulationExamples['latest-known'].expected.split('\n');
assert.equal(lookupOutput.length, 4, 'the calibration program printed four lines');
[[5, 9], [7, 9], [9, 9]].forEach(([cutoff], index) => {
  const value = latestKnown({ records: baseRecords, entity: 'sensor_A', cutoff, maximumAge: 9 }).value;
  assert.equal(lookupOutput[index], `${cutoff} ${value}`,
    `the program's line ${index} agrees with the browser model at cutoff ${cutoff}`);
  record('displayed calibration line');
});
assert.equal(lookupOutput[3], 'None', 'and the age-limited query printed None');
assert.equal(latestKnown({ records: baseRecords, entity: 'sensor_A', cutoff: 5, maximumAge: 2 }).selected, null,
  'which is what the browser model gives for the same query');
record('the displayed program agrees with the model');

/* --------------------------------------------- the timeline's own geometry */

let geometryCases = 0;
for (const records of arrivalConfigurations.slice(0, 60)) {
  for (const cutoff of [0, 1, 5, 7, 9, 12]) {
    for (const maximumAge of [0, 2, 5, 9, 12]) {
      const geometry = timelineGeometry({ records, entity: 'sensor_A', cutoff, maximumAge });
      assert.equal(geometry.lanes.length, records.length, 'one lane per record');
      const scale = geometry.scale;
      geometry.lanes.forEach(lane => {
        close(lane.eventX, scale(lane.record.event), 'the event marker sits at its own time', 1e-9);
        close(lane.availableX, scale(lane.record.available), 'and the arrival marker at its own', 1e-9);
        assert.ok(lane.availableX >= lane.eventX - 1e-9,
          'the arrival is never drawn to the left of the event it measures');
        assert.ok(lane.eventX >= geometry.labelWidth - 1e-9 && lane.eventX <= geometry.width - geometry.inset + 1e-9,
          `an event marker at ${lane.eventX} falls outside the drawn ruler`);
        assert.ok(lane.availableX >= geometry.labelWidth - 1e-9
          && lane.availableX <= geometry.width - geometry.inset + 1e-9,
          `an arrival marker at ${lane.availableX} falls outside the drawn ruler`);
        assert.ok(lane.markY > 0 && lane.markY < geometry.height, 'and every lane sits inside the frame');
        assert.equal(lane.delay, lane.record.available - lane.record.event, 'the delay is the two times apart');
      });
      assert.ok(geometry.cutoff.x >= geometry.labelWidth - 1e-9
        && geometry.cutoff.x <= geometry.width - geometry.inset + 1e-9, 'the cutoff line stays on the ruler');
      /* The defect the browser layout inspector found at 1366 px: the cutoff
         label's baseline sat at y=5 and a whole line of glyphs rose past the
         top edge of the viewBox. Every drawn baseline now has to clear its own
         ascent, checked offline as well as on the page. */
      assert.ok(geometry.cutoff.labelY - LABEL_ASCENT >= 0,
        `the cutoff label's baseline at ${geometry.cutoff.labelY} leaves its glyphs outside the viewBox`);
      assert.ok(geometry.cutoff.lineTopY >= 0 && geometry.cutoff.lineTopY <= geometry.axisY,
        'and the cutoff line starts inside the drawing');
      assert.ok(geometry.cutoff.labelY < geometry.cutoff.lineTopY,
        'with the label above the line it names');
      geometry.lanes.forEach(lane => {
        assert.ok(lane.markY - LABEL_ASCENT >= 0, 'every lane label clears the top edge too');
        assert.ok(lane.markY + LABEL_ASCENT <= geometry.height, 'and the bottom one');
        /* The second defect the browser found: the drawn identity overflowed
           its gutter and ran under the arrival marker of the record it names.
           The label must end before the ruler begins, and the ruler begins at
           the leftmost position any marker can take. */
        assert.ok(lane.drawnLabelWidth + 4 <= geometry.labelWidth,
          `the drawn lane label "${lane.drawnLabel}" is ${lane.drawnLabelWidth.toFixed(0)} units wide and the `
          + `gutter is ${geometry.labelWidth}`);
        assert.ok(lane.drawnLabelWidth < geometry.scale(geometry.domain[0]),
          'and ends before the leftmost point the ruler can place a marker at');
        assert.ok(lane.drawnLabel.length < lane.label.length,
          'the drawn label is the compact identity, not the readable one the tables carry');
      });
      assert.ok(geometry.tickLabelY + 2 <= geometry.height,
        'and the tick labels sit inside the height the figure declares');
      assert.ok(geometry.ageWindow.width >= 0, 'the age window never has negative width');
      close(geometry.ageWindow.x + geometry.ageWindow.width, geometry.cutoff.x,
        'and its right edge is the cutoff itself', 1e-9);
      assert.deepEqual(timelineLabelCollisions(geometry), [],
        `two tick labels overlap at cutoff ${cutoff}`);
      assert.ok(geometry.ticks.length >= 5, 'the ruler carries enough ticks to read');
      geometry.ticks.forEach(tick => {
        assert.ok(tick.x >= geometry.labelWidth - 1e-9 && tick.x <= geometry.width - geometry.inset + 1e-9,
          'every tick is inside the ruler');
      });
      geometryCases += 1;
    }
  }
}
assert.ok(geometryCases >= 1500, `only ${geometryCases} timeline geometries were checked`);
record('timeline geometry inside its frame');

/* The inset and label gutter are floored BEFORE any position is compared
   against them. Comparing positions against the same inset they are meant to
   guard is how a containment check became inert elsewhere. */
const defaultGeometry = timelineGeometry({ ...timelineFixture });
assert.ok(defaultGeometry.inset >= 10,
  `the timeline's inset is ${defaultGeometry.inset}, below the 10 units the end tick labels need`);
assert.ok(defaultGeometry.labelWidth >= 40,
  `the lane label gutter is ${defaultGeometry.labelWidth}, too narrow for an entity label`);
refuses(() => timelineGeometry({ ...timelineFixture, inset: 4 }), 'a timeline inset below the floor');
refuses(() => timelineGeometry({ ...timelineFixture, labelWidth: 20 }), 'a lane label gutter below the floor');
/* Reached through the LABEL, not the gutter: the four fixture identities are
   seven characters and fit any gutter the other floor allows, so a narrow
   gutter cannot exercise this guard. A longer entity name can. */
refuses(() => timelineGeometry({
  ...timelineFixture,
  records: [{ entity: 'sensor_LONGNAME', event: 12, available: 12, version: 10, value: 1 }],
}), 'an identity too long for the gutter it is drawn in');
refuses(() => timelineGeometry({ ...timelineFixture, topPadding: 20 }),
  'a top padding that puts the cutoff label outside the viewBox');
assert.ok(defaultGeometry.lanes.some(lane => Math.abs(lane.eventX - lane.availableX) > 8),
  'the default drawing really does put an event and its arrival in different places, which is what the '
  + 'figure exists to show');
record('the timeline floors');

assert.equal(recordLabel(baseRecords[2]), 'A · event 1 · v2', 'a record label names entity, event and version');
record('record labels');

/* ============================== §4-§5 · the metrics, by definition and packet */

const partition = formulationData.partition;
const validationIds = formulationData.validation.ids;
const validationTargets = formulationData.validation.targets;
nonEmpty(validationIds, partition.validationRows, 'validation identities');
nonEmpty(validationTargets, partition.validationRows, 'validation outcomes');
assert.equal(new Set(validationIds).size, validationIds.length, 'the validation identities are distinct');
assert.equal(validationTargets.reduce((sum, value) => sum + value, 0), partition.validationPositives,
  'and carry the recorded number of positive outcomes');
assert.deepEqual(validationIds, recorded.validationRows, 'the module carries the packet\'s validation rows');
assert.deepEqual(validationTargets, recorded.validationTargets, 'and the packet\'s outcomes');
record('the validation set against the packet');

const targetById = Object.fromEntries(validationIds.map((id, index) => [id, validationTargets[index]]));
const scoreSets = {
  prior: validationIds.map(() => partition.trainPrior),
  candidate: formulationData.validation.scores.candidate,
  duration: formulationData.validation.scores.duration,
};
const packetKeys = { prior: 'training_prior', candidate: 'candidate_pre_call', duration: 'unavailable_duration' };
const procedures = Object.fromEntries(formulationData.procedures.map(entry => [entry.id, entry]));
nonEmpty(Object.keys(scoreSets), 3, 'the three scored procedures');

for (const [id, scores] of Object.entries(scoreSets)) {
  const packet = recorded.results[packetKeys[id]];
  nonEmpty(scores, partition.validationRows, `${id} scores`);
  assert.deepEqual(scores, packet.probabilities.map((value, index) =>
    (id === 'prior' ? partition.trainPrior : packet.probabilities[index])),
    `${id}: the module's per-case scores are the packet's`);

  // Average precision, implemented here from the definition, against
  // scikit-learn's recorded value: a different language and a different
  // implementation, not a second call.
  /* THREE routes for each: the module under test, this file's own
     implementation, and scikit-learn's recorded output in the packet. */
  close(averagePrecision(validationTargets, scores), packet.averagePrecision,
    `${id}: the module's average precision against scikit-learn's`, 1e-12);
  close(averagePrecisionLocal(validationTargets, scores), packet.averagePrecision,
    `${id}: average precision by definition, implemented here, against scikit-learn's`, 1e-12);
  close(averagePrecisionLocal(validationTargets, scores), averagePrecision(validationTargets, scores),
    `${id}: and the two in-repo implementations against each other`, 1e-12);
  close(procedures[id].averagePrecision, packet.averagePrecision, `${id}: the module carries it unchanged`);
  // Log loss likewise.
  close(logLoss(validationTargets, scores), packet.logLoss, `${id}: the module's log loss`, 1e-12);
  close(logLossLocal(validationTargets, scores), packet.logLoss,
    `${id}: log loss by definition, implemented here`, 1e-12);
  close(procedures[id].logLoss, packet.logLoss, `${id}: the module carries it unchanged`);

  const confusion = confusionAtThreshold(validationTargets, scores, 0.5);
  assert.deepEqual(confusion.matrix, packet.confusion, `${id}: the confusion cells`);
  assert.deepEqual(confusionLocal(validationTargets, scores, 0.5), packet.confusion,
    `${id}: the confusion cells counted here, by filtering rather than by the module's switch`);
  assert.equal(confusion.correct, packet.correct, `${id}: the correct count`);
  assert.equal(confusion.matrix.flat().reduce((sum, value) => sum + value, 0), partition.validationRows,
    `${id}: every validation row lands in exactly one cell`);
  assert.deepEqual(procedures[id].confusion, packet.confusion, `${id}: the module carries the cells`);
  assert.equal(procedures[id].correct, packet.correct, `${id}: and the count`);

  // The ranking by an explicit sort against the packet's numpy lexsort, and
  // against a quadratic selection route that shares nothing with either.
  const ranking = rankByScore({ ids: validationIds, scores });
  assert.deepEqual(ranking, packet.rankedSourceRows, `${id}: the ranking against the packet`);
  const scoreById = Object.fromEntries(validationIds.map((identity, index) => [identity, scores[index]]));
  assert.deepEqual(ranking.slice(0, 120), rankBySelection(validationIds, scoreById).slice(0, 120),
    `${id}: the ranking against a repeated-maximum route`);

  const atFifty = selectionMetrics({
    order: ranking, capacity: partition.capacity, targetById, totalPositives: partition.validationPositives,
  });
  assert.equal(atFifty.positivesFound, packet.top50Positives, `${id}: positives in the selected 50`);
  close(atFifty.precision, packet.precisionAt50, `${id}: precision at 50`);
  close(atFifty.recall, packet.recallAt50, `${id}: recall at 50`);
  assert.equal(procedures[id].top50Positives, packet.top50Positives, `${id}: the module carries the count`);
  record('measured metrics by definition against the packet');
}

/* The constant baseline's closed forms. Both would fail even if this file and
   the packet agreed on a wrong number. */
close(averagePrecision(validationTargets, scoreSets.prior),
  partition.validationPositives / partition.validationRows,
  'the constant baseline\'s average precision is exactly the validation positive fraction', 0);
close(logLoss(validationTargets, scoreSets.prior),
  constantLogLoss(partition.validationPositives, partition.validationRows, partition.trainPrior),
  'and its log loss is the closed form for a constant probability', 1e-12);
assert.equal(new Set(scoreSets.prior).size, 1, 'because every one of its scores is the same number');
assert.equal(confusionAtThreshold(validationTargets, scoreSets.prior, 0.5).matrix[1][1], 0,
  'and it predicts no positive at threshold .5, so its recall of the class anyone cares about is zero');
record('the baseline\'s closed forms');

/* The comparison the lesson turns on, asserted rather than captioned. */
assert.ok(procedures.candidate.correct < procedures.prior.correct,
  'the candidate makes FEWER correct decisions at threshold .5 than the constant baseline');
assert.ok(procedures.candidate.top50Positives > 3 * procedures.prior.top50Positives,
  'while concentrating more than three times as many positives in the selected 50');
assert.ok(procedures.duration.averagePrecision > procedures.candidate.averagePrecision
  && procedures.duration.logLoss < procedures.candidate.logLoss
  && procedures.duration.correct > procedures.candidate.correct
  && procedures.duration.top50Positives > procedures.candidate.top50Positives,
  'and the unavailable-duration model is ahead on every measured number, which is the point');
record('the contrast the section exists for');

/* Perfect and reversed rankings, so average precision is exercised at both
   extremes rather than only in the middle. */
const perfectScores = validationTargets.map(target => (target === 1 ? 1 : 0));
close(averagePrecision(validationTargets, perfectScores), 1,
  'a perfect separation has average precision 1', 0);
const invertedScores = validationTargets.map(target => (target === 1 ? 0 : 1));
/* The exactly wrong ranking lands on the prevalence, not below it: the final
   threshold always admits every case, so its last step contributes the whole
   recall at a precision equal to the prevalence. A constant score reaches the
   same number by one step rather than two, which is why the baseline row and a
   deliberately inverted ranking are indistinguishable on this measure. */
close(averagePrecision(validationTargets, invertedScores),
  partition.validationPositives / partition.validationRows,
  'the exactly wrong ranking scores exactly the prevalence', 1e-15);
assert.ok(averagePrecision(validationTargets, scoreSets.candidate)
  > averagePrecision(validationTargets, invertedScores),
  'and the candidate is above that floor, so the measure is doing work');
refuses(() => averagePrecision([0, 0, 0], [0.1, 0.2, 0.3]), 'average precision with no positive case');
refuses(() => logLoss([1, 0], [1, 0.5]), 'log loss at a probability of exactly 1');
refuses(() => logLoss([1, 0], [0, 0.5]), 'and at exactly 0');
record('average precision at its extremes');

/* ===================================== §5 · capacity, swept over every value */

const rankings = Object.fromEntries(Object.entries(scoreSets)
  .map(([id, scores]) => [id, rankByScore({ ids: validationIds, scores })]));
let capacityCases = 0;
for (const [id, ranking] of Object.entries(rankings)) {
  let previousFound = 0;
  let previousSelected = new Set();
  for (let capacity = 1; capacity <= 100; capacity += 1) {
    const metrics = selectionMetrics({
      order: ranking, capacity, targetById, totalPositives: partition.validationPositives,
    });
    // Recounted here rather than trusted: the positives among the first k.
    const recount = ranking.slice(0, capacity).filter(identity => targetById[identity] === 1).length;
    assert.equal(metrics.positivesFound, recount, `${id}: the count at capacity ${capacity}`);
    close(metrics.precision, recount / capacity, `${id}: precision at ${capacity}`, 0);
    close(metrics.recall, recount / partition.validationPositives, `${id}: recall at ${capacity}`, 0);
    assert.equal(metrics.missed, partition.validationPositives - recount, `${id}: the unreached count`);
    assert.ok(recount >= previousFound, `${id}: the count fell when the capacity rose to ${capacity}`);
    assert.ok(metrics.recall >= previousFound / partition.validationPositives,
      `${id}: recall fell when the capacity rose to ${capacity}`);
    for (const identity of previousSelected) {
      assert.ok(metrics.selected.includes(identity),
        `${id}: identity ${identity} left the selected set when the capacity rose to ${capacity}`);
    }
    previousFound = recount;
    previousSelected = new Set(metrics.selected);
    capacityCases += 1;
  }
}
assert.equal(capacityCases, 300, `${capacityCases} capacity cases ran, expected 300`);
record('every capacity the control offers, for all three rankings');

/* ---------- the graded answer must be the answer the page prints ----------
 *
 * This shipped wrong. Investigation 2 graded precision at 1e-9 while printing
 * it at six decimals, so at capacity 3 the page rendered the learner's answer
 * and its own as the identical string `0.666667` and declared them different.
 * The property, asserted over every capacity the control reaches rather than
 * at the two the recorded captures happened to use: the page's printed value,
 * read back as a number, grades as correct. */
const GRADED_DIGITS = 6;
const DECLARED_TOLERANCE = 1e-9;
let printedBackCases = 0;
let wouldFailAtDeclared = 0;
for (const [id, ranking] of Object.entries(rankings)) {
  for (let capacity = 1; capacity <= 100; capacity += 1) {
    const metrics = selectionMetrics({
      order: ranking, capacity, targetById, totalPositives: partition.validationPositives,
    });
    const printedBack = Number(metrics.precision.toFixed(GRADED_DIGITS));
    const allowance = gradingAllowance({ declared: DECLARED_TOLERANCE, digits: GRADED_DIGITS });
    assert.ok(Math.abs(printedBack - metrics.precision) <= allowance,
      `${id}: at capacity ${capacity} the page prints ${printedBack} and grades against `
      + `${metrics.precision} with an allowance of ${allowance}; a learner copying the page's own number `
      + 'would be told it is outside');
    if (Math.abs(printedBack - metrics.precision) > DECLARED_TOLERANCE) wouldFailAtDeclared += 1;
    printedBackCases += 1;
  }
}
assert.equal(printedBackCases, 300, `${printedBackCases} printed-back cases ran, expected 300`);
/* The floor that makes this non-vacuous: if the declared tolerance were used
   raw, a large share of these would fail. Asserting that keeps the guard from
   quietly becoming a tautology if the display precision ever changes. */
assert.ok(wouldFailAtDeclared > 150,
  `only ${wouldFailAtDeclared} of ${printedBackCases} cases would fail at the declared ${DECLARED_TOLERANCE}; `
  + 'if that number reaches zero this assertion has stopped testing anything');
close(gradingAllowance({ declared: 0, digits: 6 }), 5e-7, 'six printed decimals allow half a unit in the last', 0);
close(gradingAllowance({ declared: 1e-3, digits: 6 }), 1e-3, 'a looser declared tolerance is kept', 0);
close(gradingAllowance({ declared: 0, digits: 0 }), 0.5, 'and an integer display allows half a unit');
refuses(() => displayTolerance(-1), 'a negative display precision');
refuses(() => gradingAllowance({ declared: -1, digits: 6 }), 'a negative declared tolerance');
record('the graded allowance is never tighter than the printed precision');

refuses(() => selectionMetrics({
  order: rankings.candidate, capacity: 0, targetById, totalPositives: partition.validationPositives,
}), 'a capacity of zero');
refuses(() => selectionMetrics({
  order: rankings.candidate, capacity: partition.validationRows + 1, targetById,
  totalPositives: partition.validationPositives,
}), 'a capacity beyond the evaluated set');
refuses(() => selectionMetrics({
  order: [1, 1, 2], capacity: 2, targetById: { 1: 0, 2: 1 }, totalPositives: 1,
}), 'a duplicated identity in one action set');
refuses(() => selectionMetrics({
  order: rankings.candidate, capacity: 5, targetById, totalPositives: 0,
}), 'a recall denominator of zero');

/* Capacity 1 and the whole evaluated set: the two ends of the control. */
const atOne = selectionMetrics({
  order: rankings.candidate, capacity: 1, targetById, totalPositives: partition.validationPositives,
});
assert.equal(atOne.selected.length, 1, 'a capacity of 1 selects one opportunity');
assert.ok(atOne.precision === 0 || atOne.precision === 1, 'whose precision is 0 or 1 and nothing between');
const atAll = selectionMetrics({
  order: rankings.candidate, capacity: partition.validationRows, targetById,
  totalPositives: partition.validationPositives,
});
close(atAll.recall, 1, 'selecting everything reaches every positive outcome', 0);
close(atAll.precision, partition.validationPositives / partition.validationRows,
  'and its precision is the prevalence', 0);
assert.equal(atAll.missed, 0, 'with nothing unreached');
record('the two ends of the capacity control');

/* The recorded practice answers. */
const candidateAtTwentyFive = selectionMetrics({
  order: rankings.candidate, capacity: 25, targetById, totalPositives: partition.validationPositives,
});
assert.equal(candidateAtTwentyFive.positivesFound, recorded.policyFixtures.top25.positives,
  'the candidate\'s first 25 contain the recorded number of positives');
assert.equal(candidateAtTwentyFive.positivesFound, 12, 'which is 12');
close(candidateAtTwentyFive.precision, 0.48, 'so precision at 25 is .48', 0);
close(candidateAtTwentyFive.recall, 12 / partition.validationPositives, 'and recall is 12/90', 0);
assert.deepEqual(candidateAtTwentyFive.selected, recorded.policyFixtures.top25.ids,
  'and the identities are the recorded ones');
assert.deepEqual(candidateAtTwentyFive.selected, formulationData.policyFixtures.top25Ids,
  'which the module carries unchanged');
const candidateAtFifty = selectionMetrics({
  order: rankings.candidate, capacity: partition.capacity, targetById,
  totalPositives: partition.validationPositives,
});
assert.ok(candidateAtTwentyFive.precision > candidateAtFifty.precision,
  'precision RISES when the capacity halves, on this ranking');
assert.ok(candidateAtTwentyFive.recall < candidateAtFifty.recall, 'while recall falls');
assert.ok(candidateAtTwentyFive.selected.every(identity => candidateAtFifty.selected.includes(identity)),
  'because the top 25 are a subset of the top 50, which is why recall could not have risen');
record('the practice-4 answer');

/* ------------------------------------------- the exchange, swept exhaustively */

let exchangeCases = 0;
let exchangeSeen = { down: 0, up: 0, flat: 0 };
for (const capacity of [25, 50]) {
  const order = rankings.candidate;
  const before = selectionMetrics({
    order, capacity, targetById, totalPositives: partition.validationPositives,
  });
  const selected = order.slice(0, capacity);
  const unselected = order.slice(capacity);
  nonEmpty(selected, capacity, `selected identities at capacity ${capacity}`);
  nonEmpty(unselected, partition.validationRows - capacity, `unselected identities at capacity ${capacity}`);
  for (const removeId of selected) {
    for (const addId of unselected) {
      const swapped = swapSelection({ order, capacity, removeId, addId });
      const after = selectionMetrics({
        order: swapped, capacity, targetById, totalPositives: partition.validationPositives,
      });
      const expected = before.positivesFound - targetById[removeId] + targetById[addId];
      assert.equal(after.positivesFound, expected,
        `exchanging ${removeId} for ${addId} at capacity ${capacity} gave ${after.positivesFound}, not ${expected}`);
      assert.equal(after.selected.length, capacity, 'and the capacity is unchanged');
      assert.equal(new Set(after.selected).size, capacity, 'with no identity selected twice');
      const change = membershipChange(before.selected, after.selected);
      assert.deepEqual(change.entered, [addId], 'exactly one identity entered');
      assert.deepEqual(change.left, [removeId], 'and exactly one left');
      assert.equal(change.held, capacity - 1, 'the rest held');
      const movement = movementOf(before.precision, after.precision);
      if (expected < before.positivesFound) exchangeSeen.down += 1;
      else if (expected > before.positivesFound) exchangeSeen.up += 1;
      else exchangeSeen.flat += 1;
      assert.equal(movement.outcome,
        expected < before.positivesFound ? 'falls' : expected > before.positivesFound ? 'rises' : 'unchanged',
        'and the graded direction follows the count');
      exchangeCases += 1;
    }
  }
}
assert.ok(exchangeCases >= 50000, `only ${exchangeCases} exchanges were swept`);
assert.ok(exchangeSeen.down > 0 && exchangeSeen.up > 0 && exchangeSeen.flat > 0,
  `the sweep saw ${JSON.stringify(exchangeSeen)}; all three graded directions must occur or the grading rule `
  + 'was only ever exercised in one branch');
record('every exchange at two capacities');

/* The recorded exchange, and the membership null. */
const recordedSwap = formulationData.policyFixtures.swap;
assert.equal(targetById[recordedSwap.removeId], 1, 'the recorded exchange drops a subscription');
assert.equal(targetById[recordedSwap.addId], 0, 'and takes one that did not subscribe');
const afterRecordedSwap = selectionMetrics({
  order: swapSelection({
    order: rankings.candidate, capacity: 25, removeId: recordedSwap.removeId, addId: recordedSwap.addId,
  }),
  capacity: 25, targetById, totalPositives: partition.validationPositives,
});
assert.equal(afterRecordedSwap.positivesFound, recordedSwap.positives, 'costing exactly one');
assert.equal(afterRecordedSwap.positivesFound, 11, 'which is 11 of 25');
record('the recorded exchange');

let reversalCases = 0;
for (const [id, ranking] of Object.entries(rankings)) {
  for (let capacity = 1; capacity <= 100; capacity += 1) {
    const before = selectionMetrics({
      order: ranking, capacity, targetById, totalPositives: partition.validationPositives,
    });
    const after = selectionMetrics({
      order: reverseSelected({ order: ranking, capacity }), capacity, targetById,
      totalPositives: partition.validationPositives,
    });
    assert.equal(after.positivesFound, before.positivesFound,
      `${id}: reversing the selected block at capacity ${capacity} changed the count`);
    close(after.precision, before.precision, `${id}: or the precision at ${capacity}`, 0);
    close(after.recall, before.recall, `${id}: or the recall at ${capacity}`, 0);
    assert.equal(movementOf(before.precision, after.precision).outcome, 'unchanged',
      `${id}: and the graded direction is unchanged at capacity ${capacity}`);
    const change = membershipChange(before.selected, after.selected);
    assert.deepEqual(change.entered, [], 'nothing entered');
    assert.deepEqual(change.left, [], 'nothing left');
    reversalCases += 1;
  }
}
assert.equal(reversalCases, 300, `${reversalCases} reversal cases ran, expected 300`);
record('the membership null at every capacity');

refuses(() => swapSelection({ order: rankings.candidate, capacity: 25, removeId: rankings.candidate[0], addId: rankings.candidate[1] }),
  'exchanging two already-selected opportunities');
refuses(() => swapSelection({ order: rankings.candidate, capacity: 25, removeId: rankings.candidate[30], addId: rankings.candidate[40] }),
  'exchanging two unselected opportunities');
refuses(() => swapSelection({ order: rankings.candidate, capacity: 25, removeId: rankings.candidate[0], addId: rankings.candidate[0] }),
  'exchanging an opportunity with itself');
refuses(() => swapSelection({ order: rankings.candidate, capacity: 25, removeId: -1, addId: rankings.candidate[30] }),
  'an identity that is not in the ranking');

/* policyOrder replays a whole edit history rather than mutating state. */
const replayed = policyOrder({
  baseOrder: rankings.candidate, capacity: 25,
  swaps: [[recordedSwap.removeId, recordedSwap.addId]], reversed: false,
});
assert.deepEqual(replayed.slice(0, 25).sort((a, b) => a - b),
  swapSelection({ order: rankings.candidate, capacity: 25, removeId: recordedSwap.removeId, addId: recordedSwap.addId })
    .slice(0, 25).sort((a, b) => a - b),
  'replaying one exchange gives the same selected set as applying it');
const replayedTwice = policyOrder({
  baseOrder: rankings.candidate, capacity: 25,
  swaps: [[recordedSwap.removeId, recordedSwap.addId]], reversed: true,
});
assert.deepEqual([...replayedTwice.slice(0, 25)].reverse(), replayed.slice(0, 25),
  'and a reversal afterwards reverses exactly the selected block');
assert.deepEqual(policyOrder({ baseOrder: rankings.candidate, capacity: 25 }), rankings.candidate,
  'an empty edit history is the model ranking itself');
record('policy replay');

/* ---------------------------------------- the unchanged rule, as an iff */

let movementCases = 0;
for (const difference of [0, unchangedTolerance, -unchangedTolerance, unchangedTolerance / 2,
  unchangedTolerance * 2, -unchangedTolerance * 2, 1e-6, -1e-6, 0.5, -0.5]) {
  for (const base of [0, 0.4, 0.48, 1, -0.3]) {
    const after = base + difference;
    /* The REALISED difference, not the nominal one: 1 + 1e-12 and 1 differ by
       slightly more than 1e-12 in double arithmetic, and a test that used the
       nominal value would be asserting something the function was never asked
       to do. The iff is about the numbers the function actually receives. */
    const realised = after - base;
    const outcome = movementOf(base, after).outcome;
    const expected = Math.abs(realised) <= unchangedTolerance ? 'unchanged' : realised > 0 ? 'rises' : 'falls';
    assert.equal(outcome, expected,
      `movementOf(${base}, ${after}) gave ${outcome}; unchanged iff the realised difference `
      + `${realised} is within ${unchangedTolerance}`);
    movementCases += 1;
  }
}
/* The boundary is tested from zero, where the difference is exactly the value
   written rather than the nearest double to it. At 0.4 the spacing is about
   5.5e-17 and a "just past the tolerance" step rounds back onto it, which would
   make the test assert something other than inclusivity. */
assert.equal(movementOf(0, unchangedTolerance).outcome, 'unchanged',
  'the tolerance boundary is inclusive');
assert.equal(movementOf(0, -unchangedTolerance).outcome, 'unchanged', 'on both sides');
assert.equal(movementOf(0, unchangedTolerance * 2).outcome, 'rises', 'and just past it is not');
assert.equal(movementOf(0, -unchangedTolerance * 2).outcome, 'falls', 'in either direction');
assert.equal(movementOf(0, -0).outcome, 'unchanged', 'signed zeros are the same number');
assert.equal(movementOf(0.48, 0.48).difference, 0, 'an identical pair reports a zero difference');
refuses(() => movementOf(Number.NaN, 0.4), 'a movement from a value that is not a number');
refuses(() => movementOf(0.4, Number.POSITIVE_INFINITY), 'and to an infinite one');
assert.ok(movementCases >= 50, `only ${movementCases} movement cases ran`);
record('the unchanged rule as an if and only if');

/* ============================================= §4 · the cost threshold figure */

let thresholdCases = 0;
for (let falsePositiveCost = 1; falsePositiveCost <= 12; falsePositiveCost += 1) {
  for (let falseNegativeCost = 1; falseNegativeCost <= 12; falseNegativeCost += 1) {
    const threshold = costThreshold(falsePositiveCost, falseNegativeCost);
    close(threshold, falsePositiveCost / (falsePositiveCost + falseNegativeCost), 'the threshold closed form', 0);
    // Independent route: the probability at which the two expected losses are
    // equal, found by bisection on the loss difference rather than by the ratio.
    let low = 0;
    let high = 1;
    for (let step = 0; step < 80; step += 1) {
      const middle = (low + high) / 2;
      const { acting, waiting } = expectedLosses(middle, falsePositiveCost, falseNegativeCost);
      if (acting > waiting) low = middle; else high = middle;
    }
    close((low + high) / 2, threshold, 'the threshold found by bisection on the loss difference', 1e-9);
    const below = expectedLosses(Math.max(threshold - 0.05, 0), falsePositiveCost, falseNegativeCost);
    const above = expectedLosses(Math.min(threshold + 0.05, 1), falsePositiveCost, falseNegativeCost);
    assert.equal(below.preferred, 'wait', 'below the threshold the cheaper choice is to wait');
    assert.equal(above.preferred, 'act', 'and above it, to act');
    const geometry = thresholdPlotGeometry({ falsePositiveCost, falseNegativeCost });
    close(geometry.threshold, threshold, 'the drawn crossing is the closed form');
    close(geometry.crossing.x, geometry.x(threshold), 'placed by the figure\'s own scale', 1e-9);
    // The crossing must lie ON both drawn lines, not merely near them.
    const onLine = (line, x) => line.from.y
      + ((x - line.from.x) / (line.to.x - line.from.x)) * (line.to.y - line.from.y);
    close(onLine(geometry.actingLine, geometry.crossing.x), geometry.crossing.y,
      'and on the acting line the figure draws', 1e-9);
    close(onLine(geometry.waitingLine, geometry.crossing.x), geometry.crossing.y,
      'and on the waiting line', 1e-9);
    [geometry.actingLine.from, geometry.actingLine.to, geometry.waitingLine.from, geometry.waitingLine.to,
      geometry.crossing].forEach(point => {
      assert.ok(point.x >= geometry.padding.left - 1e-9 && point.x <= geometry.width - geometry.padding.right + 1e-9,
        `a drawn point at x ${point.x} falls outside the plot frame`);
      assert.ok(point.y >= geometry.padding.top - 1e-9
        && point.y <= geometry.height - geometry.padding.bottom + 1e-9,
        `a drawn point at y ${point.y} falls outside the plot frame`);
    });
    thresholdCases += 1;
  }
}
assert.equal(thresholdCases, 144, `${thresholdCases} cost pairs were checked, expected 144`);
close(costThreshold(5, 5), 0.5, 'equal costs put the threshold at .5, which is the only case that does', 0);
close(costThreshold(3, 9), 0.25, 'practice 6\'s threshold is 3/12', 0);
close(expectedLosses(0.2, 3, 9).acting, 2.4, 'acting costs 2.4 at p = .2', 1e-12);
close(expectedLosses(0.2, 3, 9).waiting, 1.8, 'and waiting 1.8', 1e-12);
assert.equal(expectedLosses(0.2, 3, 9).preferred, 'wait', 'so the cheaper choice is not to act');
assert.equal(expectedLosses(costThreshold(3, 9), 3, 9).preferred, 'indifferent',
  'and exactly at the threshold the two are equal');
refuses(() => costThreshold(0, 9), 'a zero cost');
refuses(() => expectedLosses(1.2, 3, 9), 'a probability above 1');
record('the cost threshold and its figure');

/* ============================================ §8 · stocking and the uplift */

const demands = [10, 20];
const probabilities = [0.5, 0.5];
close(stockingCost({ stock: 10, demands, probabilities, underageCost: 3, overageCost: 1 }), 15,
  'stocking 10 costs 15 in expectation', 1e-12);
close(stockingCost({ stock: 20, demands, probabilities, underageCost: 3, overageCost: 1 }), 5,
  'and stocking 20 costs 5', 1e-12);
close(stockingCost({ stock: 15, demands, probabilities, underageCost: 3, overageCost: 1 }), 10,
  'while stocking the mean costs 10, between the two');
assert.equal(optimalStock({ demands, probabilities, underageCost: 3, overageCost: 1 }).stock, 20,
  'the critical fraction selects the higher level');
close(criticalQuantile(3, 1), 0.75, 'which is the .75 quantile', 0);
/* Both formulas are one ratio of two costs, and the roles are swapped: the
   acting threshold puts the cost of acting wrongly on top, the stocking
   fraction puts the cost of running short on top. With underage 3 and overage
   1 that is 1/4 against 3/4 -- complements, not equals. The figure says so and
   this is the assertion behind it. */
close(costThreshold(1, 3), 0.25, 'the acting threshold with the overage as the false-positive cost', 0);
assert.notEqual(criticalQuantile(3, 1), costThreshold(1, 3),
  'the stocking fraction and the acting threshold are NOT the same number for one cost pair');
close(criticalQuantile(3, 1) + costThreshold(1, 3), 1, 'they are complements', 1e-12);
assert.ok(/complements, not equals/.test(
  fs.readFileSync('src/learn/components/lesson-labs/FormulationFigures.jsx', 'utf8')),
  'and the figure says so on the page rather than only here');
refuses(() => stockingCost({ stock: 10, demands, probabilities: [0.5, 0.4], underageCost: 3, overageCost: 1 }),
  'demand probabilities that do not sum to 1');
record('the stocking decision');

const uplift = upliftGeometry({});
assert.equal(uplift.rows.length, upliftGroups.length, 'two constructed groups');
close(uplift.rows[0].increment, 0.05, 'group A gains .05 from being called', 1e-12);
close(uplift.rows[1].increment, 0.35, 'and group B gains .35', 1e-12);
assert.equal(uplift.leaderByCalledProbability, 'A', 'ranking by probability under the action favours A');
assert.equal(uplift.leaderByIncrement, 'B', 'ranking by increment favours B');
assert.equal(uplift.reverses, true, 'so the ordering reverses, which is the figure\'s whole claim');
uplift.rows.forEach(row => {
  close(row.called.endX, uplift.scale(row.called.value), 'each bar ends at its own value', 1e-9);
  close(row.notCalled.endX, uplift.scale(row.notCalled.value), 'on the same shared scale', 1e-9);
  assert.ok(row.called.endX <= uplift.width - uplift.padding.right + 1e-9, 'and stays inside the frame');
  close(Math.abs(row.bracket.toX - row.bracket.fromX),
    Math.abs(uplift.scale(row.called.value) - uplift.scale(row.notCalled.value)),
    'the difference bracket spans exactly the difference', 1e-9);
  record('uplift bar');
});
assert.equal(uplift.scale.domain[0], 0, 'the probability axis starts at 0');
assert.equal(uplift.scale.domain[1], 1, 'and ends at 1, so the bars are comparable');
record('the propensity-against-impact figure');

/* ========================================= §§1-2, 5-6 · the other geometries */

const flow = flowGeometry({});
assert.equal(flow.boxes.length, flowStages.length, 'five stages');
assert.equal(flow.arrows.length, flowStages.length - 1, 'and four arrows between them');
assert.equal(flow.arrowsFromOutcomeIntoPrediction, 0,
  'no arrow runs from the observed outcome forward into a prediction');
assert.equal(flow.feedback.fromId, 'outcome', 'the feedback channel starts at the outcome');
assert.equal(flow.feedback.toId, 'records', 'and returns to the record store, not to the prediction');
assert.match(flow.feedback.note, /after the outcome exists/, 'and says when that is legitimate');
flow.boxes.forEach((box, index) => {
  assert.ok(box.y >= 0 && box.y + box.height <= flow.height, `box ${index} is inside the frame`);
  assert.ok(box.x >= 0 && box.x + box.width <= flow.width, 'horizontally too');
  assert.ok(box.title.length * 5.7 < box.width,
    `the label "${box.title}" needs about ${(box.title.length * 5.7).toFixed(0)} units and its box is ${box.width}`);
  assert.ok(box.badge.length * 5.1 < box.width, `and so does the badge "${box.badge}"`);
  if (index > 0) {
    assert.ok(box.y > flow.boxes[index - 1].y + flow.boxes[index - 1].height,
      'and sits below the one before it with a gap');
  }
  record('flow box');
});
flow.feedback.points.forEach(point => {
  assert.ok(point.x >= 0 && point.x <= flow.width, 'the feedback channel stays inside the frame');
  assert.ok(point.y >= 0 && point.y <= flow.height, 'vertically too');
});
assert.ok(flow.feedback.points.some(point => point.x > flow.boxes[0].x + flow.boxes[0].width),
  'and runs outside the boxes rather than through them');
record('the decision-flow figure');

const lanes = splitLaneGeometry({});
assert.equal(lanes.views.length, 2, 'two allocations of the same rows');
lanes.views.forEach(view => {
  assert.equal(view.rows, parcelFixture.parcels.length * parcelFixture.hours.length, 'nine observations each');
  assert.equal(view.trainRows + view.validationRows, view.rows, 'every row lands on exactly one side');
  const seen = new Set(view.cells.map(cell => cell.id));
  assert.equal(seen.size, view.rows, 'and appears exactly once');
  view.cells.forEach(cell => {
    assert.ok(cell.x >= 0 && cell.x + cell.width <= lanes.width + 1e-9, 'each cell is inside the frame');
    assert.ok(cell.y >= 0 && cell.y + cell.height <= lanes.height + 1e-9, 'vertically too');
  });
  record('split allocation');
});
assert.ok(lanes.views[0].identityOverlap > 0,
  'the row-random allocation puts at least one parcel identity on both sides');
assert.equal(lanes.views[1].identityOverlap, 0, 'and the identity allocation puts none');
assert.equal(lanes.views[1].distinctTrainParcels + lanes.views[1].distinctValidationParcels,
  parcelFixture.parcels.length, 'with the identities partitioned rather than shared');
record('the unit-split figure');

const nested = nestedPartitionGeometry({ partition });
assert.equal(nested.first.segments.length, 2, 'the first bar has two segments');
assert.equal(nested.second.segments.length, 2, 'and so does the second');
close(nested.first.segments[0].rows + nested.first.segments[1].rows, partition.sourceRows,
  'the first bar accounts for every source row', 0);
close(nested.second.segments[0].rows + nested.second.segments[1].rows, partition.developmentRows,
  'and the second for every development row', 0);
close(nested.first.segments[0].endX, nested.second.segments[1].endX,
  'both bars share one scale, so the second ends where the development segment does', 1e-9);
nested.first.segments.concat(nested.second.segments).forEach(segment => {
  close(segment.width, (segment.rows / partition.sourceRows) * nested.width,
    `the ${segment.id} segment's width encodes its row count`, 1e-9);
  assert.ok(segment.x >= -1e-9 && segment.x + segment.width <= nested.width + 1e-9,
    'and stays inside the bar');
  record('partition segment');
});
assert.deepEqual(nested.unscoredSegments, ['reserved'], 'exactly one segment is never scored');
assert.equal(nested.first.segments.find(segment => segment.id === 'reserved').scored, false,
  'and it is the reserved one');
refuses(() => partitionGeometry({
  scaleTotal: 100, groups: [{ id: 'a', rows: 80 }, { id: 'b', rows: 40 }],
}), 'segments that overflow their own scale');
record('the fitting-boundary bars');

const availability = availabilityAxisGeometry({});
assert.equal(availability.rows.length, AVAILABILITY_FEATURES.length, 'four feature families');
assert.equal(availability.afterCount, 1, 'exactly one of them is unavailable before the call');
assert.equal(availability.rows.find(row => row.side === 'after').label, 'duration',
  'and it is the final call duration');
availability.rows.forEach(row => {
  assert.ok(row.x >= 0 && row.x + row.width <= availability.width + 1e-9, 'every family bar is inside the frame');
  if (row.side === 'before') {
    close(row.x + row.width, availability.cutoffX, 'a pre-call family ends exactly at the cutoff', 1e-9);
    assert.equal(row.crossesCutoff, false, 'and does not cross it');
  } else {
    close(row.x, availability.cutoffX, 'the post-call family starts exactly at the cutoff', 1e-9);
    assert.equal(row.crossesCutoff, true, 'and is marked as crossing it');
  }
  record('availability family');
});
record('the availability axis');

for (const metric of Object.keys(metricAxes)) {
  const rows = formulationData.procedures.map(procedure => ({
    id: procedure.id, short: procedure.id, averagePrecision: procedure.averagePrecision,
    logLoss: procedure.logLoss, precisionAt50: procedure.precisionAt50,
  }));
  const geometry = metricBarGeometry({ rows, metric });
  assert.equal(geometry.bars.length, 3, `${metric}: three bars`);
  geometry.bars.forEach(bar => {
    close(bar.endX, geometry.scale(bar.value), `${metric}: the ${bar.id} bar ends at its own value`, 1e-9);
    close(bar.width, geometry.scale(bar.value) - geometry.scale(geometry.axis.domain[0]),
      `${metric}: and its length encodes that value`, 1e-9);
    assert.ok(bar.endX <= geometry.width - geometry.padding.right + 1e-9,
      `${metric}: the ${bar.id} bar runs past the plot frame`);
    assert.ok(bar.labelX < geometry.padding.left, `${metric}: the name label sits in its own gutter`);
    assert.ok(bar.valueX > bar.endX, `${metric}: and the value label sits past the bar it belongs to`);
    assert.ok(bar.y >= geometry.padding.top && bar.y + geometry.barHeight <= geometry.height,
      `${metric}: and every bar is inside the frame`);
  });
  // The bars must ORDER as their values do, or the picture disagrees with the table.
  const byValue = [...geometry.bars].sort((left, right) => right.value - left.value);
  const byLength = [...geometry.bars].sort((left, right) => right.width - left.width);
  assert.deepEqual(byLength.map(bar => bar.id), byValue.map(bar => bar.id),
    `${metric}: the drawn lengths order differently from the values they encode`);
  record('metric bar geometry');
}
refuses(() => metricBarGeometry({
  rows: [{ id: 'x', short: 'x', averagePrecision: 0.9 }], metric: 'averagePrecision',
}), 'a value outside the drawn axis');
refuses(() => metricBarGeometry({ rows: [{ id: 'x', short: 'x' }], metric: 'nothingLikeThis' }),
  'a metric with no declared axis');
record('the results figure');

for (const procedure of formulationData.procedures) {
  const strip = selectedStripGeometry({ capacity: partition.capacity, positives: procedure.top50Positives });
  assert.equal(strip.cells.length, partition.capacity, 'one cell per selected opportunity');
  assert.equal(strip.cells.filter(cell => cell.positive).length, procedure.top50Positives,
    `${procedure.id}: the marked cells are the recorded count`);
  strip.cells.forEach(cell => {
    assert.ok(cell.x >= -1e-9 && cell.x + cell.width <= strip.width + 1e-9, 'inside the strip');
  });
  record('selected-case strip');
}
refuses(() => selectedStripGeometry({ capacity: 50, positives: 51 }), 'more positives than cells');
refuses(() => selectedStripGeometry({ capacity: 400, positives: 1 }), 'cells too narrow to read');

const lineage = lineageGeometry({});
assert.equal(lineage.rows.length, lineagePaths.length, 'three information paths');
assert.equal(new Set(lineage.rows.map(row => row.kind)).size, 3, 'each of a different kind');
lineage.rows.forEach(row => {
  assert.equal(row.arrows.length, row.nodes.length - 1, 'one arrow between each pair of nodes');
  row.nodes.forEach(node => {
    assert.ok(node.x >= 0 && node.x + node.width <= lineage.width + 1e-9, 'each node is inside the frame');
    node.lines.forEach(line => {
      assert.ok(line.length * 4.4 < node.width,
        `the lineage label "${line}" needs about ${(line.length * 4.4).toFixed(0)} units and its node is `
        + `${node.width.toFixed(0)}`);
    });
  });
  row.arrows.forEach(arrow => {
    assert.ok(arrow.to.x > arrow.from.x, 'and every arrow runs left to right');
    assert.equal(arrow.from.y, arrow.to.y, 'horizontally');
  });
  assert.ok(row.repair.length > 40, 'each path carries a repair rather than a label');
  record('lineage path');
});
refuses(() => lineageGeometry({
  paths: [{ id: 'x', title: 'x', role: 'x', kind: 'x', repair: 'x', nodes: [['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g']] }],
}), 'more nodes than the row can draw');
record('the lineage figure');

/* ======================================== §§3-4 · the small exact fixtures */

const policyA = policyOutcome({ ...supportFixture, policy: supportFixture.policies[0] });
const policyB = policyOutcome({ ...supportFixture, policy: supportFixture.policies[1] });
assert.equal(policyA.correct, 9, 'policy A is correct on 9 of 12');
assert.equal(policyA.cost, 21, 'at a cost of 21');
assert.equal(policyB.correct, 11, 'policy B is correct on 11 of 12');
assert.equal(policyB.cost, 2, 'at a cost of 2');
assert.equal(policyB.withinCapacity, false, 'but exceeds a capacity of 3');
assert.equal(policyA.withinCapacity, true, 'while A does not, which is why the comparison is not a ranking');
assert.ok(policyB.correct > policyA.correct && policyB.cost < policyA.cost,
  'so B is better on both accuracy and cost and still infeasible: the point of the exercise');
refuses(() => policyOutcome({
  ...supportFixture, policy: { id: 'C', escalated: 2, urgentEscalated: 5 },
}), 'escalating more urgent requests than exist');
record('the escalation policies');

const majority = majorityBaseline({ negatives: 90, positives: 10 });
close(majority.accuracy, 0.9, 'the majority baseline reaches 90% accuracy', 0);
assert.equal(majority.positiveRecall, 0, 'with zero recall of the positive class');
close(majority.prevalence, 0.1, 'at a prevalence of 10%', 0);
refuses(() => majorityBaseline({ negatives: 0, positives: 0 }), 'a prevalence over no cases');
record('the prevalence example');

/* linearScale, which every figure is built on. */
const scale = linearScale({ domain: [0, 1], range: [10, 110] });
close(scale(0), 10, 'a scale maps its domain start to its range start', 0);
close(scale(1), 110, 'and its end to its end', 0);
close(scale(0.25), 35, 'linearly between', 1e-12);
close(scale.invert(35), 0.25, 'and inverts', 1e-12);
refuses(() => linearScale({ domain: [1, 1], range: [0, 10] }), 'a scale with no domain width');
refuses(() => scale(Number.NaN), 'a scale input that is not a number');
refuses(() => checkFinite(Number.POSITIVE_INFINITY, 'x'), 'an infinite value');
record('the shared scale');

/* ========================== the manuscript and the lesson body agree with all this */

const manuscriptClaims = [
  ['4,119-row', /4,119-row random subset/],
  ['451 positive rows', /451 rows have the positive label/],
  ['the reserved rows', /reserve 824 rows without scoring them/],
  ['the fit split', /use 2,471 to fit and 824 for the stated diagnostic/],
  ['the cutoff-5 answer', /the appropriate value in this fixture is therefore \*\*10\*\*/],
  ['the moved arrival', /arrives at time 4 instead, the answer becomes 20/],
  ['the top-25 answer', /contains 12 positives in its first 25/],
  ['the threshold', /The threshold is \\\(3\/12=\.25\\\)/],
];
nonEmpty(manuscriptClaims, 8, 'manuscript claims checked');
manuscriptClaims.forEach(([label, pattern]) => {
  assert.match(manuscript, pattern, `the manuscript still states ${label}`);
  record('manuscript claim');
});
const manuscriptTable = manuscript.match(/\| Training-prior baseline \| \.109223 \| \.344889 \| 734\/824 \| 6 \|/);
assert.ok(manuscriptTable, 'the manuscript\'s results table still carries the baseline row');
assert.match(manuscript, /\| Candidate recorded-feature model \| \.253440 \| \.336527 \| 733\/824 \| 20 \|/,
  'and the candidate row');
assert.match(manuscript, /\| Same feature families plus final duration \| \.461700 \| \.264639 \| 738\/824 \| 26 \|/,
  'and the duration row');
[['prior', 0.109223, 0.344889, 734, 6], ['candidate', 0.253440, 0.336527, 733, 20],
  ['duration', 0.461700, 0.264639, 738, 26]].forEach(([id, ap, loss, correct, found]) => {
  close(Number(procedures[id].averagePrecision.toFixed(6)), ap, `${id}: the module rounds to the manuscript's AP`, 0);
  close(Number(procedures[id].logLoss.toFixed(6)), loss, `${id}: and its log loss`, 0);
  assert.equal(procedures[id].correct, correct, `${id}: and its correct count`);
  assert.equal(procedures[id].top50Positives, found, `${id}: and its top-50 count`);
  record('manuscript results row');
});

/* The lesson body must not print an answer the labs grade, and must carry the
   things the page promises. */
assert.ok(lessonBody.includes('formulation-lesson'), 'the body carries the lesson root class');
/* Comments are stripped first. This stylesheet's own explanation of the hazard
   quotes the forbidden selector, and a scan that flagged the explanation would
   be a scan nobody could keep green -- which is how a guard gets deleted rather
   than fixed. */
const stylesheet = fs.readFileSync('src/learn/components/lesson-labs/formulation-labs.css', 'utf8');
const stylesheetRules = stylesheet.replace(/\/\*[\s\S]*?\*\//g, '');
assert.ok(!/(^|[\s,>+~])svg\s*(\{|,)/m.test(stylesheetRules),
  'the stylesheet has a bare svg selector, which would also match KaTeX\'s radical SVGs');
assert.ok(stylesheetRules.includes('svg.form-svg'),
  'and the layout rule is scoped to this lesson\'s own SVG class');
assert.ok(/[^.]svg\.form-svg/.test(stylesheetRules), 'attached to the element, not to a container');
record('the scoped SVG layout rule');
assert.ok(lessonBody.includes('reserved'), 'and states that rows are reserved');
assert.ok(/Both investigations ask for a recorded prediction/.test(lessonBody),
  'the intro promises what the investigations keep');
/* Every procedure carries an availability contract in the data module. A
   contract that is recorded but never rendered is a contract nobody reads, and
   on this page that is the one thing the whole lesson is about. */
const figuresBody = fs.readFileSync('src/learn/components/lesson-labs/FormulationFigures.jsx', 'utf8');
const labsBody = fs.readFileSync('src/learn/components/lesson-labs/FormulationLabs.jsx', 'utf8');
assert.ok(figuresBody.includes('procedure.availability'),
  'the results figure renders each procedure\'s availability contract, not only its score');
assert.ok(labsBody.includes('.availability'),
  'and the capacity investigation renders the contract of whichever ranking is selected');
formulationData.procedures.forEach(procedure => {
  assert.ok(procedure.availability.length > 60,
    `${procedure.id}: its availability contract is a sentence rather than a label`);
  record('availability contract');
});
assert.match(procedures.duration.availability, /does not exist until the call has ended/,
  'and the duration model\'s says plainly why it cannot be an input');
record('the lesson body');

/* The examples module's recorded programs. */
assert.equal(formulationExamples['latest-known'].source.blockIndex, 0,
  'the calibration program is the manuscript\'s first fenced block');
assert.equal(formulationExamples.experiment.source.blockIndex, 1, 'and the experiment its second');
assert.ok(formulationExamples['latest-known'].code.includes('def latest_known('),
  'the displayed calibration program really defines the function it is named for');
assert.ok(formulationExamples.experiment.code.includes('def make_model('),
  'and the experiment defines its pipeline builder');
assert.ok(manuscript.includes(formulationExamples['latest-known'].code),
  'the displayed calibration code appears verbatim in the frozen manuscript');
assert.ok(manuscript.includes(formulationExamples.experiment.code),
  'and so does the displayed experiment');
assert.equal(formulationExamples.calculations.downloadOnly, true,
  'the complete calculation program is offered as a download rather than displayed');
assert.equal(formulationExamples.calculations.producesBytes,
  fs.statSync(`${packetDirectory}/calculated-inputs.json`).size,
  'and reproduces a results file of exactly the packet\'s size');
['numpy', 'pandas', 'scikit-learn'].forEach(name => {
  const resolved = formulationExamples.experiment.environment[name];
  assert.ok(resolved, `the examples module records a ${name} version`);
  const key = name === 'scikit-learn' ? 'scikitLearn' : name;
  assert.equal(formulationData.software[key], resolved,
    `the data module's ${name} version matches the one the programs ran on`);
  assert.ok(lessonBody.includes(`formulationData.software.${key}`)
    || lessonBody.includes(`environment['${name}']`) || lessonBody.includes(`environment.${name}`),
    `and the page reads that version rather than printing a literal for ${name}`);
  record('recorded runtime version');
});
record('the displayed programs');

/* ================================================================ evidence */

const sources = [
  'src/learn/data/formulation-models.js',
  'src/learn/data/formulation-data.js',
  'src/learn/data/formulation-examples.js',
  'src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx',
  'src/learn/data/curriculum/blueprints/ml-problem-formulation-baselines-data-leakage.js',
  'src/learn/components/lesson-labs/FormulationShared.jsx',
  'src/learn/components/lesson-labs/FormulationLabs.jsx',
  'src/learn/components/lesson-labs/FormulationFigures.jsx',
  'src/learn/components/lesson-labs/formulation-labs.css',
  'public/learn-assets/problem-formulation/bank-additional.csv',
  'public/learn-assets/problem-formulation/ATTRIBUTION.txt',
  'public/learn-assets/problem-formulation/bank-marketing-variable-description.txt',
  'public/learn-assets/problem-formulation/formulation-calculations.py',
];
// Unfiltered on purpose: silently dropping a declared source that no longer
// exists and still writing `passed: true` is how a deleted file passes.
const missingSources = sources.filter(file => !fs.existsSync(file));
assert.deepEqual(missingSources, [], `declared source files are missing: ${missingSources.join(', ')}`);

const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
/* Floors sit just below the current values, so a lost section fails the run
   while a legitimate addition does not. A counter that is reported but never
   floored is decoration: deleting every record() call would still print PASS. */
assert(total >= 138, `only ${total} grouped checks ran; the suite has lost coverage`);
assert(Object.keys(counts).length >= 60, `only ${Object.keys(counts).length} groups ran`);
assert(timelineCases >= 50000, `only ${timelineCases} as-known cases ran`);
assert(exchangeCases >= 50000, `only ${exchangeCases} exchange cases ran`);
assert(capacityCases === 300, `only ${capacityCases} capacity cases ran`);
assert(reversalCases === 300, `only ${reversalCases} reversal cases ran`);
assert(geometryCases >= 1500, `only ${geometryCases} timeline geometries ran`);
assert(thresholdCases === 144, `only ${thresholdCases} cost pairs ran`);
assert(nullCases >= 2000, `only ${nullCases} null cases ran`);

const evidence = {
  checkedAt: new Date().toISOString(),
  sourceHashes: Object.fromEntries(sources.map(file => [file, hash(file)])),
  packetCalculatedInputsSha256: hash(`${packetDirectory}/calculated-inputs.json`),
  packetManuscriptSha256: hash(`${packetDirectory}/lesson.md`),
  verifierHash: hash('scripts/verify-formulation-models.mjs'),
  counts,
  totalGroupedChecks: total,
  asKnownCasesSwept: timelineCases,
  asKnownSelectionsSeen: timelineSelected,
  asKnownEmptyResultsSeen: timelineEmpty,
  asKnownNullCasesSwept: nullCases,
  timelineGeometriesSwept: geometryCases,
  capacityCasesSwept: capacityCases,
  exchangeCasesSwept: exchangeCases,
  exchangeDirectionsSeen: exchangeSeen,
  reversalCasesSwept: reversalCases,
  movementCasesSwept: movementCases,
  costPairsSwept: thresholdCases,
  scope: 'The browser problem-formulation models against the content packet\'s calculated-inputs.json, the '
    + 'frozen manuscript\'s stated values and printed program output, and a second derivation for every claim '
    + 'a figure draws. The as-known selection is computed by a second route that sorts a zero-padded composite '
    + 'key and takes the last element, and swept over EVERY cutoff, EVERY maximum age, both entities and 150 '
    + 'arrival configurations; both outcomes -- a selection and a missing calibration -- are shown to occur, '
    + 'and the leakage property itself (the selected record never arrives after the cutoff) is asserted on '
    + 'every case. Average precision, log loss, the confusion cells and the ranking are implemented here from '
    + 'their definitions and checked against scikit-learn\'s recorded outputs, a different language and a '
    + 'different implementation; the ranking is additionally checked against a quadratic repeated-maximum '
    + 'route, and the constant baseline against its closed forms. The capacity rule is checked at every '
    + 'capacity the control offers for all three rankings, with monotonicity and the nesting of successive '
    + 'selected sets; every exchange of one selected for one unselected opportunity at capacities 25 and 50 is '
    + 'applied and its arithmetic checked, with all three graded directions shown to occur; and the reversal '
    + 'null is checked at every capacity. The unchanged rule is asserted as an if and only if over a '
    + 'degenerate grid including signed zeros and the exact tolerance boundary. Every drawn coordinate -- '
    + 'timeline markers, cutoff, age window, flow boxes, split cells, partition segments, availability bars, '
    + 'metric bars, selected-case strips, lineage nodes, uplift bars and the cost crossing -- is checked '
    + 'against its own scale and required to stay inside its frame, with the crossing required to lie on both '
    + 'drawn lines and the bar order required to match the value order.',
  limitations: [
    'The fitted probabilities are the packet\'s recorded measurements. They are regenerated from the served '
      + 'dataset by scripts/verify-formulation-data.py; this file checks that the browser module carries them '
      + 'unchanged and that everything computed FROM them is right.',
    'Displayed program output is executed separately by scripts/verify-formulation-examples.py; this file '
      + 'checks that what those programs printed agrees with what the browser models compute.',
    'Label-collision checking here uses a nominal character width. The real glyph geometry is measured on the '
      + 'rendered page by scripts/verify-formulation-browser.cjs.',
    'The as-known sweep varies arrivals, cutoff, age and entity. Event times and version numbers are fixed in '
      + 'the investigation, so they are fixed here; a future control that edits them would need version '
      + 'semantics defined first and this sweep extended.',
    'Rendering, interaction, visual layout and independent review are separate steps and are not claimed here.',
  ],
  passed: true,
};
/* `--no-evidence` lets an independent reviewer re-run this without writing to
 * the record they are reviewing. This is the point at which the provisional
 * record above is replaced, and the only point at which `passed` becomes true. */
if (keepEvidence) {
  fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
}
console.log(`PASS: ${total} grouped problem-formulation model checks across ${Object.keys(counts).length} groups, `
  + `including ${timelineCases.toLocaleString('en-US')} as-known cases over every cutoff, age and entity `
  + `(${timelineSelected.toLocaleString('en-US')} selections, ${timelineEmpty.toLocaleString('en-US')} missing `
  + `calibrations), ${nullCases.toLocaleString('en-US')} null cases, `
  + `${geometryCases.toLocaleString('en-US')} timeline geometries, ${capacityCases} capacity cases, `
  + `${exchangeCases.toLocaleString('en-US')} exchanges `
  + `(${exchangeSeen.down} falling, ${exchangeSeen.up} rising, ${exchangeSeen.flat} unchanged), `
  + `${reversalCases} membership nulls and ${thresholdCases} cost pairs.`);
