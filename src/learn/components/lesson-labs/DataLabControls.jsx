export function DataStepControls({ step, count, setStep, reset }) {
  return <div className="data-step-controls">
    <button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
    <span>Step {step + 1} of {count}</span>
    <button type="button" disabled={step === count - 1} onClick={() => setStep(step + 1)}>Next step</button>
    <button type="button" onClick={reset}>Reset</button>
  </div>;
}

