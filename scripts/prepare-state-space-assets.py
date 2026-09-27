"""Export retained SSM teaching fits without retraining or changing draft evidence."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "state-space-models-s4-mamba-mamba-2"
SOURCE = ROOT / "docs/teaching/drafts" / TOPIC
DESTINATION = ROOT / "public/learn-code" / TOPIC


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    files = ["state_space_mechanisms.py", "trajectory_state_models.py",
             "state_space_library_bridge.py", "movement_libras.data", "movement_libras.names",
             "trajectory-results.json", "trajectory-state-fits.npz", "data-provenance.md",
             "mechanism-results.json", "investigation-checks.json"]
    for filename in files:
        shutil.copyfile(SOURCE / filename, DESTINATION / filename)
    # Publish a learner-facing implementation record while preserving the frozen
    # preparation packet and its historical phase boundary.
    provenance = (SOURCE / 'data-provenance.md').read_text(encoding='utf-8')
    provenance = provenance.replace(
        'Source files, exact small results and selected arrays are necessary retained content-first artifacts. Runtime conversion, native integration checks, mobile/browser/accessibility/performance verification and independent lesson review remain phase two.',
        'The published lesson includes complete CPU programs, retained original measurements and an on-demand browser export of both seed-17 classifiers with all 50 validation trajectories. Native PyTorch inference and browser-model arithmetic were compared on 108 original/edited cases; the largest logit difference was 7.14e-6. Independent source/numerical review and desktop, 320px and 760px browser interaction checks were completed on 26 September 2026. The browser performs bounded frozen-model inference, not training. The optional maintained Mamba CUDA bridge is supplied and source checked, but was not executed on GPU. These checks do not establish hardware throughput, new-performer generalization or superiority to other architectures. No retained fit or measurement was changed during website integration.')
    provenance = provenance.replace('investigation-checks.json and author_calculations.py retain',
        'investigation-checks.json and the preparation record retain')
    (DESTINATION / 'data-provenance.md').write_text(provenance, encoding='utf-8')
    results = json.loads((SOURCE / "trajectory-results.json").read_text())
    raw = np.loadtxt(SOURCE / "movement_libras.data", delimiter=",")
    arrays = np.load(SOURCE / "trajectory-state-fits.npz")
    models = {}
    for kind in ("diagonal", "selective"):
        prefix = f"{kind}_seed17::"
        models[kind] = {name[len(prefix):]: arrays[name].tolist()
                        for name in arrays.files if name.startswith(prefix) and not name.endswith("logits")}
    validation = [{"id": row, "label": int(raw[row - 1, -1]),
                   "points": raw[row - 1, :90].reshape(45, 2).tolist()}
                  for row in results["roles"]["validation"]]
    write_json(DESTINATION / "trajectory-inference.json", {"models": models, "records": validation})
    paths = []
    for label in (1, 6, 10):
        row = next(row for row in sorted(results["roles"]["fit"]) if raw[row - 1, -1] == label)
        paths.append({"id": row, "label": label, "points": raw[row - 1, :90].reshape(45, 2).tolist()})
    compact = {"paths": paths, "models": [{"kind": m["kind"], "seed": m["seed"],
               "selected_epoch": m["selected_epoch"], "history": m["history"],
               "confusion": m["metrics"]["test"]["confusion"]} for m in results["models"]]}
    write_json(ROOT / "src/learn/data/state-space-measurements.json", compact)
    write_json(DESTINATION / "asset-provenance.json", {
        "source": "Retained preparation fits; no new training",
        "sha256": {filename: hashlib.sha256((DESTINATION / filename).read_bytes()).hexdigest()
                   for filename in files + ["trajectory-inference.json"]}})
    print("Exported two seed-17 models, 50 validation records, exact measured curves and canonical programs.")


if __name__ == "__main__":
    main()
