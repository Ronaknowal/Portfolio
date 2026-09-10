"""Independent real-NumPy oracle for bounded browser teaching models."""
import json
import sys
import numpy as np

request = json.load(sys.stdin)
X = np.array([[18, 20], [24, 26], [30, 32]], dtype=np.float64)
answer = {"runtime": {"python": sys.version.split()[0], "numpy": np.__version__}}
answer["selections"] = []
for selection in request["selections"]:
    result = np.asarray(eval(selection["code"], {"X": X, "np": np}))
    answer["selections"].append({"shape": list(result.shape), "values": result.ravel().tolist(),
                                 "source": [X.ravel().tolist().index(v) for v in result.ravel()]})
answer["memory"] = []
for case in request["memory"]:
    raw = X.copy()
    picked = (raw[:, 0] if case["kind"] == "view" else raw[:, 0].copy()
              if case["kind"] == "copy" else raw[[0, 1, 2], 0])
    if case["index"] is not None:
        picked[case["index"]] = 99
    answer["memory"].append({"original": raw.ravel().tolist(), "picked": picked.tolist(),
                            "shares": bool(np.shares_memory(raw, picked)),
                            "offsets": [int(i * picked.strides[0]) for i in range(picked.size)]})
answer["broadcast"] = []
for case in request["broadcast"]:
    a = np.asarray(case["a"]).reshape(case["aShape"])
    b = np.asarray(case["b"]).reshape(case["bShape"])
    try:
        result = a - b
        ai = np.broadcast_to(np.arange(a.size).reshape(a.shape), result.shape)
        bi = np.broadcast_to(np.arange(b.size).reshape(b.shape), result.shape)
        answer["broadcast"].append({"valid": True, "shape": list(result.shape),
                                    "values": result.ravel().tolist(),
                                    "aIndices": ai.ravel().tolist(), "bIndices": bi.ravel().tolist()})
    except ValueError:
        answer["broadcast"].append({"valid": False})
answer["reductions"] = []
for case in request["reductions"]:
    result = X.mean(axis=case["axis"], keepdims=case["keepdims"])
    answer["reductions"].append({"shape": list(result.shape), "values": result.ravel().tolist()})
print(json.dumps(answer))
