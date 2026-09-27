import { frequencies, rotate } from './positional-encoding-models.js';
import { dot } from './sequence-tensor-operations.js';

// All vectors use adjacent pairs externally. P changes storage to half-split
// order; P is its own inverse for this four-coordinate teaching head.
export function rotaryConventionTrace(query, key, queryId, keyId, base, pairs = 1) {
  const permutation = [0, 2, 1, 3];
  const freq = frequencies(4, base);
  const packed = permutation.map(i => query[i]);
  const halfRotated = [...packed];
  for (let j = 0; j < 2; j++) {
    const c = Math.cos(queryId * freq[j]), s = Math.sin(queryId * freq[j]);
    halfRotated[j] = packed[j] * c - packed[j + 2] * s;
    halfRotated[j + 2] = packed[j] * s + packed[j + 2] * c;
  }
  const rq = rotate(query, queryId, base), rk = rotate(key, keyId, base);
  const contributions = [0, 1].map(pair => dot(
    (pair < pairs ? rq : query).slice(pair * 2, pair * 2 + 2),
    (pair < pairs ? rk : key).slice(pair * 2, pair * 2 + 2)));
  return { permutation, packed, halfRotated, restored: permutation.map(i => halfRotated[i]), adjacent: rq, contributions };
}

export function xposPairTrace(queryId = 512, keyId = 0) {
  const zeta = 2 / 7, scale = 512;
  // A declared first-pair example: q=k=[1,0], frequency one radian/ID.
  const queryAmplitude = zeta ** (queryId / scale), keyAmplitude = zeta ** (-keyId / scale);
  const query = [Math.cos(queryId), Math.sin(queryId)];
  const key = [Math.cos(keyId), Math.sin(keyId)];
  return { query, key, queryAmplitude, keyAmplitude,
    scaledQuery: query.map(v => v * queryAmplitude), scaledKey: key.map(v => v * keyAmplitude),
    product: queryAmplitude * keyAmplitude, relativeAmplitude: zeta ** ((queryId - keyId) / scale),
    rotaryDot: dot(query, key), scaledDot: queryAmplitude * keyAmplitude * dot(query, key) };
}
