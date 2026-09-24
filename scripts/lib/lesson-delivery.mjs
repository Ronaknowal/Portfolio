import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

export const deliveryLedgerPath = 'docs/teaching/lesson-delivery-progress.json';
const statuses = new Set(['not-started', 'in-progress', 'complete']);
const modes = new Set(['full', 'content-first']);

function validateFiles(files, description) {
  if (!files || !Object.keys(files).length) throw new Error(`${description} needs a nonempty file/hash map.`);
  for (const [filename, digest] of Object.entries(files)) {
    if (path.isAbsolute(filename) || /^[A-Za-z]:/.test(filename) || filename.split(/[\\/]/).includes('..')) {
      throw new Error(`${description} must use repository-relative paths.`);
    }
    if (!/^[a-f0-9]{64}$/.test(digest)) throw new Error(`${description}: invalid SHA256 for ${filename}.`);
  }
}

export function validateDeliveryLedger(ledger, knownIds) {
  if (ledger.schemaVersion !== 1 || !ledger.topics || Array.isArray(ledger.topics)) {
    throw new Error('Expected lesson delivery ledger schemaVersion 1 and topics keyed by stable ID.');
  }
  for (const [id, entry] of Object.entries(ledger.topics)) {
    if (knownIds && !knownIds.has(id)) throw new Error(`Unknown delivery topic: ${id}`);
    if (!Number.isInteger(entry.revision) || entry.revision < 1 || !modes.has(entry.deliveryMode)) {
      throw new Error(`${id}: invalid revision or delivery mode.`);
    }
    for (const phase of ['content', 'implementation']) {
      if (!statuses.has(entry[phase]?.status)) throw new Error(`${id}: invalid ${phase} status.`);
    }
    if (entry.implementation.status !== 'not-started' && entry.content.status !== 'complete') {
      throw new Error(`${id}: implementation cannot start before content is complete.`);
    }
    if (typeof entry.record !== 'string' || !entry.record) throw new Error(`${id}: missing handoff record.`);
    if (entry.legacyReview === true) {
      if (entry.content.status !== 'complete' || entry.implementation.status !== 'complete') {
        throw new Error(`${id}: legacyReview represents a completed historical revision only.`);
      }
    } else {
      if (entry.content.status === 'complete') {
        validateFiles(entry.content.files, `${id} content checkpoint`);
        for (const field of ['manuscript', 'visualSpecifications']) {
          if (typeof entry.content[field] !== 'string' || !Object.hasOwn(entry.content.files, entry.content[field])) {
            throw new Error(`${id}: content checkpoint must identify and hash its ${field}.`);
          }
        }
      }
      if (entry.implementation.status === 'complete') validateFiles(entry.implementation.reviewedFiles, `${id} reviewed implementation`);
    }
  }
}

export function filesMatch(root, files) {
  return Boolean(files && Object.keys(files).length && Object.entries(files).every(([filename, expected]) => {
    const absolute = path.resolve(root, filename);
    const relative = path.relative(root, absolute);
    if (relative.startsWith('..') || path.isAbsolute(relative) || !fs.existsSync(absolute)) return false;
    return fs.statSync(absolute).isFile() && createHash('sha256').update(fs.readFileSync(absolute)).digest('hex') === expected;
  }));
}

// Historical review checks remain owned by their original source-bound ledgers.
// The new ledger owns delivery phases, not a second copy of old evidence.
export function getDeliveryState(root, entry, legacyImplementationCurrent = false) {
  if (!entry) return { content: 'not-started', implementation: 'not-started', canFinish: false };
  const contentCurrent = entry.legacyReview ? legacyImplementationCurrent
    : entry.content.status === 'complete' && filesMatch(root, entry.content.files);
  const implementationCurrent = entry.legacyReview ? legacyImplementationCurrent
    : entry.implementation.status === 'complete' && filesMatch(root, entry.implementation.reviewedFiles);
  return {
    revision: entry.revision,
    deliveryMode: entry.deliveryMode,
    content: entry.content.status === 'complete' && !contentCurrent ? 'stale' : entry.content.status,
    implementation: entry.implementation.status === 'complete' && (!implementationCurrent || !contentCurrent)
      ? 'stale' : entry.implementation.status,
    recordedContent: entry.content.status,
    recordedImplementation: entry.implementation.status,
    canFinish: entry.content.status === 'complete' && contentCurrent,
    record: entry.record,
    contentFiles: entry.content.files ? Object.keys(entry.content.files) : undefined,
    nextAction: entry.nextAction,
    pendingRevision: entry.pendingRevision,
  };
}

// This is a read-only preflight. Request authorization and stage changes are
// recorded by the author; merely running the command never completes a phase.
export function assertDeliveryRequest(work, state) {
  if (!['full', 'content', 'finish'].includes(work)) throw new Error('Use --work full, content or finish.');
  if (work === 'finish' && !state.canFinish) {
    throw new Error('Finish requires a complete, current content checkpoint. Generate or reconcile the written content and visual/lab specifications first; publication or a blueprint alone is insufficient.');
  }
}

// A content-only draft leaves an older published version's review intact.
// Once rewriting/implementation is active, legacy allowlists cannot certify it.
export function getImplementationReviewOverride(entry, state) {
  if (!entry || entry.legacyReview || (entry.deliveryMode === 'content-first' && entry.implementation.status === 'not-started')) return null;
  return state.implementation === 'complete'
    ? { teachingReview: 'implementation-reviewed-user-acceptance-pending', teachingReviewRecord: entry.record }
    : { teachingReview: 'individual-review-required', teachingReviewRecord: undefined };
}
