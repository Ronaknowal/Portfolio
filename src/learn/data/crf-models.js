/** Exact, bounded models for the CRF investigations. Factors are dimensionless. */
export const LABELS = ['A', 'B'];
export const BIO_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG'];
export const initialFactors = [3, 1, 1, 2, 1, 4, 1, 1];
export const factorPresets = {
  Initial: initialFactors,
  'Pair reverses the choice': [3, 1, 6, 2, 1, 4, 1, 1],
  'Strong input wins': [3, 1, 12, 2, 1, 4, 1, 1],
  'No interactions': [3, 1, 6, 2, 1, 1, 1, 1],
  'Four-way tie': Array(8).fill(1),
};
export function validFactors(values, minimum = .125, maximum = 16) {
  return values.every(value => value !== '' && Number.isFinite(Number(value)) && Number(value) >= minimum && Number(value) <= maximum);
}
export function chainDistribution(values) {
  if (values.length !== 8 || !validFactors(values)) throw new RangeError('Use eight finite factors from 0.125 to 16.');
  const factors = values.map(Number);
  const paths = LABELS.flatMap((first, i) => LABELS.map((second, j) => ({
    name: first + second, i, j, factors: [factors[i], factors[4 + 2 * i + j], factors[2 + j]],
    mass: factors[i] * factors[4 + 2 * i + j] * factors[2 + j],
  })));
  const partition = paths.reduce((sum, path) => sum + path.mass, 0);
  const largest = Math.max(...paths.map(path => path.mass));
  const winners = paths.filter(path => Math.abs(path.mass - largest) <= 1e-12 * largest).map(path => path.name);
  const incoming = LABELS.map((_, j) => paths.filter(path => path.j === j));
  return { factors, paths: paths.map(path => ({ ...path, probability: path.mass / partition })), partition, winners,
    forward: incoming.map(entries => entries.reduce((sum, path) => sum + path.mass, 0)),
    maximum: incoming.map(entries => Math.max(...entries.map(path => path.mass))),
    parents: incoming.map(entries => {
      const largestIncoming = Math.max(...entries.map(path => path.mass));
      return entries.filter(path => Math.abs(path.mass - largestIncoming) <= 1e-12 * largestIncoming).map(path => LABELS[path.i]).join(', ');
    }),
    nodes: [0, 1].map(position => LABELS.map((_, label) => paths.filter(path => (position ? path.j : path.i) === label).reduce((sum, path) => sum + path.mass, 0) / partition)),
  };
}
export function independentDistribution(values) { return chainDistribution([...values.slice(0, 4), 1, 1, 1, 1]); }
export function sameWinners(left, right) { return left.winners.join(',') === right.winners.join(','); }
export function normalizationBranches(p, a, b) {
  if (!validFactors([p], .05, .95) || !validFactors([a, b], .001, 10)) throw new RangeError('Use p in [0.05, 0.95] and positive compatibility factors in [0.001, 10].');
  const prior = Number(p), first = Number(a), second = Number(b);
  const masses = [prior * first, (1 - prior) * second];
  const partition = masses[0] + masses[1];
  const global = masses.map(mass => mass / partition);
  const delta = global[0] - prior;
  return { local: [prior, 1 - prior], global, masses, partition, direction: Math.abs(delta) <= 1e-10 ? 'unchanged' : delta > 0 ? 'increase' : 'decrease' };
}
export function legalBio(previous, current) {
  if (!BIO_LABELS.includes(current) || (previous !== 'START' && !BIO_LABELS.includes(previous))) return false;
  return !current.startsWith('I-') || previous === `B-${current.slice(2)}` || previous === current;
}
export function legalBioPath(path) { return path.every((label, index) => legalBio(index ? path[index - 1] : 'START', label)); }
/** Independent general forward normalizer, also supporting hard constraints. */
export function logPartition(emissions, transitions, startAllowed = emissions[0].map(() => true)) {
  const sumExp = values => {
    const maximum = Math.max(...values);
    return maximum === -Infinity ? -Infinity : maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
  };
  let message = emissions[0].map((value, index) => startAllowed[index] ? value : -Infinity);
  for (const row of emissions.slice(1)) message = row.map((value, j) => value + sumExp(message.map((prefix, i) => prefix + transitions[i][j])));
  return sumExp(message);
}
