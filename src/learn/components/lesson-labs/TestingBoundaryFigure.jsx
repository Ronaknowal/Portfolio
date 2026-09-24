import "./python-mechanism-figures.css";

export function TestingBoundaryFigure() {
  return <figure className="pmf-figure" data-python-figure="test-boundary">
    <figcaption><strong>The test's starting point determines what it exercises.</strong><span>These are the two concrete examples below, with the same numeric readings.</span></figcaption>
    <div className="pmf-test-route"><strong>Controlled reader</strong><div className="pmf-test-source pmf-test-substitute"><small>Mock supplies text</small><code>{'"18\\n24\\n"'}</code></div><span className="pmf-link">→</span><div className="pmf-test-calculation"><small>REAL CALCULATION</small><code>[18.0, 24.0] → 21.0</code></div><p>Check result, call count and reader errors. File paths, opening and decoding are bypassed.</p></div>
    <div className="pmf-test-route"><strong>Temporary-file integration</strong><div className="pmf-test-source"><small>Actual UTF-8 file</small><code>{'18\\n\\n24\\n'}</code></div><span className="pmf-link">→</span><div className="pmf-test-real"><small>REAL FILE BOUNDARY</small><span>path → open → decode</span><span>iterate lines → close</span></div><span className="pmf-link">→</span><div className="pmf-test-calculation"><small>REAL CALCULATION</small><code>[18.0, 24.0] → 21.0</code></div><p>Exercise file I/O and blank-line parsing; then check the calculated mean and temporary-directory removal.</p></div>
    <p className="pmf-caption">Both can return 21.0 while testing different boundaries. The second is a local file integration check, not evidence about a network service, every encoding or all deployment platforms.</p>
  </figure>;
}
