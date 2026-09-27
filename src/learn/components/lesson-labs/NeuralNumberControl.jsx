import { useEffect, useId, useState } from 'react';
import './neural-number-control.css';

// Preserve the error's footprint on blur: removing it during pointerdown can
// move a following button before pointerup and swallow the learner's click.
export function NeuralNumber({ label, value, onChange, min, max, step = 'any', integer = false, range = true }) {
  const id = useId();
  const [draft, setDraft] = useState(null);
  const [reserveMessage, setReserveMessage] = useState(false);
  const invalid = draft !== null;
  useEffect(() => { setDraft(null); setReserveMessage(false); }, [value]);
  const commit = number => { setDraft(null); setReserveMessage(false); onChange(number); };
  return <div className={`neural-number neural-number-control${range ? '' : ' neural-number-control--numeric'}`}>
    <label htmlFor={`${id}-number`}>{label}</label>
    {range && <input id={`${id}-range`} type="range" aria-label={`${label} slider`}
      min={min} max={max} step={integer && step === 'any' ? 1 : step} value={value}
      onChange={event => commit(Number(event.target.value))} />}
    <input id={`${id}-number`} type="number" min={min} max={max}
      step={integer && step === 'any' ? 1 : step} value={draft ?? value}
      aria-invalid={invalid} aria-describedby={invalid ? `${id}-error` : undefined}
      onBlur={() => setDraft(null)}
      onChange={event => {
        const text = event.target.value, number = Number(text);
        if (text.trim() && Number.isFinite(number) && number >= min && number <= max && (!integer || Number.isInteger(number))) commit(number);
        else { setDraft(text); setReserveMessage(true); }
      }} />
    {reserveMessage && <small id={`${id}-error`} aria-hidden={!invalid}
      style={{ visibility: invalid ? 'visible' : 'hidden' }}>
      Enter {integer ? 'a whole number from ' : ''}{min} to {max}. The views retain the last valid value: {value}.
    </small>}
  </div>;
}
