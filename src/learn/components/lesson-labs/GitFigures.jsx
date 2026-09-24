import "./workflow-figures.css";

export function GitRemotePicture({ state, step, localChange }) {
  return <figure className="wv-figure wv-lab-figure" data-visual="git-repository-boundary">
    <div className="wv-remote-server"><span>Shared repository · another repository</span><strong>main → <code>{state.shared}</code></strong><small>{step === 1 ? 'The colleague has pushed here' : 'This branch belongs to the shared repository'}</small></div>
    <div className="wv-fetch-link" data-active={step === 3}><span aria-hidden="true">↓</span><code>git fetch origin</code><small>Copies objects and updates your recorded view</small></div>
    <div className="wv-local-repository"><strong>Your local repository</strong><div className="wv-git-local-flow"><div><span>Remote-tracking reference</span><code>origin/main → {state.tracking}</code><small>Stored on your machine</small></div><div className="wv-integrate-link" data-active={step === 4}><span aria-hidden="true">→</span><code>merge --ff-only</code><small>{step === 4 && localChange ? 'Refused: histories diverged' : 'A separate integration decision'}</small></div><div><span>Your checked-out branch</span><code>main → {state.local}</code><small>HEAD names main</small></div></div><div className="wv-working-file"><span>Working file on your disk</span><code>{state.working}</code></div></div>
    <figcaption>Repository boundaries matter: both main and origin/main in the lower region are local references. A colleague's push changes the upper repository first; fetch refreshes origin/main, and integration is what may move your main. Arrows label operations, not continuous synchronization.</figcaption>
  </figure>;
}
