export const argumentCases = {
  spaces: { value: 'run alpha.csv', files: ['run alpha.csv'], split: ['run', 'alpha.csv'] },
  wildcard: { value: '*.csv', files: ['a.csv', 'run alpha.csv'], split: ['a.csv', 'run alpha.csv'] },
  empty: { value: '', files: [], split: [] },
};

export function argumentTrace(caseId, quoted) {
  const fixture = argumentCases[caseId];
  if (!fixture) throw new Error('Unknown argument fixture');
  return { value: fixture.value, argv: quoted ? [fixture.value] : fixture.split,
    phases: ['Read command syntax', `Expand $value to ${JSON.stringify(fixture.value)}`,
      quoted ? 'Double quotes retain this one argument, even when empty' : 'Apply default IFS splitting, then filename expansion',
      'Launch the command with the resulting argument boundaries'] };
}

export function pipelineStatus(statuses, pipefail) {
  if (!statuses.length || statuses.some(n => !Number.isInteger(n) || n < 0 || n > 255)) throw new Error('Invalid statuses');
  return pipefail ? [...statuses].reverse().find(n => n !== 0) ?? 0 : statuses.at(-1);
}

export function publicationTrace(failure = false) {
  return [
    { step: 'Validate input and create owned workspace', staged: 'absent', visible: 'previous complete report', status: 'not started' },
    { step: 'Write staged file inside destination directory', staged: 'partial new report', visible: 'previous complete report', status: 'running' },
    { step: failure ? 'Producer exits 4: reject its partial output' : 'Producer exits 0: accept complete output', staged: failure ? 'partial new report' : 'complete new report', visible: 'previous complete report', status: failure ? 'failed (4)' : 'success (0)' },
    { step: failure ? 'Skip rename; cleanup removes staged file' : 'Rename staged file to destination; cleanup removes workspace', staged: 'absent', visible: failure ? 'previous complete report' : 'complete new report', status: failure ? 'failure returned' : 'success returned' },
  ];
}
