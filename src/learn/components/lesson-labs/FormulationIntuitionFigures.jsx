import './formulation-intuition.css';

export function SelectiveLabelsFigure() {
  return <figure className="formulation-intuition" data-figure="selected-label-support">
    <div className="formulation-selection-total">1,000 produced parts</div>
    <div className="formulation-selection-branches">
      <div><strong>100 inspected</strong><span>Old rule sends these to measurement</span><span>20 defective · 80 not defective</span><span>Measured defect rate: 20%</span></div>
      <div><strong>900 not inspected</strong><span>No precise defect labels</span><span>Defective: unknown · not defective: unknown</span><span>Missing labels are not zero defects</span></div>
    </div>
    <figcaption>Constructed counts. The observed 20/100 describes the selected branch. Without further assumptions, the whole production batch could have as few as 20 defective parts or as many as 920: between 2% and 92%. The width of that range comes from absent evidence about the 900, not a poor classifier. Random inspection of some previously uninspected cases can supply information the old rule omitted.</figcaption>
  </figure>;
}
