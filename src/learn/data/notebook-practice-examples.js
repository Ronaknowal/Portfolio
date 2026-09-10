// Independently runnable notebookPractice examples; verified code/output pairs.
export const notebookPracticeExamples = {
notebookTransfer:{filename:'replay_measurements.py',code:`from hashlib import sha256
from pathlib import Path
import json
import platform
import tempfile

def analyse(raw, *, offset_ms):
    values = json.loads(raw)
    # This exercise uses the stated, trusted finite numeric fixture.
    adjusted = [value - offset_ms for value in values]
    return {"count": len(values), "mean_ms": sum(adjusted) / len(adjusted)}

def fingerprint(raw, offset_ms, source):
    ingredients = {
        "data_sha256": sha256(raw).hexdigest(),
        "offset_ms": offset_ms,
        "source_sha256": sha256(source).hexdigest(),
        "python": platform.python_version(),
    }
    encoded = json.dumps(ingredients, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest(), ingredients

if __name__ == "__main__":
    raw = b"[10,20,30]"
    offset = 5
    source = Path(__file__).read_bytes()
    identity, inputs = fingerprint(raw, offset, source)
    result = analyse(raw, offset_ms=offset)
    manifest = {"identity": identity, "inputs": inputs, "status": "complete", "result": result}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "raw.json").write_bytes(raw)
        (root / "source.py").write_bytes(source)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        replay_id, _ = fingerprint((root / "raw.json").read_bytes(),
                                  saved["inputs"]["offset_ms"],
                                  (root / "source.py").read_bytes())
        print("identity matches:", replay_id == saved["identity"])
        print("replay:", analyse(raw, offset_ms=saved["inputs"]["offset_ms"]))
        alternate = b"[9,20,31]"
        new_id, _ = fingerprint(alternate, offset, source)
        print("same result:", analyse(alternate, offset_ms=offset) == result)
        print("different identity:", new_id != identity)`,output:"identity matches: True\nreplay: {'count': 3, 'mean_ms': 15.0}\nsame result: True\ndifferent identity: True"}
};
