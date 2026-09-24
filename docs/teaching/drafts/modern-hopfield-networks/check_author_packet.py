"""Bounded manuscript calculations and retained-artifact checks; no production review."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch
from torch import nn
from digit_memory import load_roles, read_memory
from associative_memory import store_binary, binary_recall

ROOT = Path(__file__).parent
REPO = ROOT.parents[3]
torch.set_num_threads(1)
lesson = (ROOT / "lesson.md").read_text(encoding="utf8")
spec = (ROOT / "visual-specifications.md").read_text(encoding="utf8")
blocks = re.findall(r"~~~python\n(.*?)\n~~~", lesson, re.S)
assert len(blocks) == 1
namespace = {}
exec(compile(blocks[0], "lesson-inline-read", "exec"), namespace)
roles, metadata = load_roles()
arrays = np.load(ROOT / "digit-memory-fits.npz")
errors = {}
with torch.no_grad():
    for seed in (17, 41):
        projection = nn.Linear(64, 16, bias=False)
        projection.weight.copy_(torch.from_numpy(arrays[f"seed{seed}_projection"]))
        inputs = (roles["validation"][0][:5], *roles["memory"])
        a = namespace["label_read"](*inputs, projection)
        b = read_memory(*inputs, 16., projection)
        errors[str(seed)] = max(float((x-y).abs().max()) for x,y in zip(a,b))
        assert errors[str(seed)] < 1e-6
        assert (projection(roles["memory"][0]).norm(dim=-1) > 0).all()

training = np.loadtxt(ROOT/"optdigits.tra", delimiter=",", dtype=int)
testing = np.loadtxt(ROOT/"optdigits.tes", delimiter=",", dtype=int)
all_features = np.concatenate((training[:,:64], testing[:,:64]))
assert len(np.unique(all_features, axis=0)) == 5620
role_ids = metadata["roles_training_source_ids"]
assert [len(role_ids[name]) for name in ("memory", "fit_queries", "validation")] == [200,800,300]
assert len(set(sum(role_ids.values(),[]))) == 1300
for name,count in (("memory",20),("fit_queries",80),("validation",30)):
    assert np.all(np.bincount(roles[name][1].numpy(),minlength=10)==count)
assert all(np.isfinite(arrays[name]).all() for name in arrays.files)

def check_confusions(value):
    if isinstance(value,dict):
        if "confusion" in value:
            matrix = np.array(value["confusion"])
            assert matrix.sum() == value["count"]
            assert matrix.sum()-np.trace(matrix) == value["errors"]
        for child in value.values():
            check_confusions(child)
    elif isinstance(value,list):
        for child in value:
            check_confusions(child)
check_confusions(json.loads((ROOT/"digit-results.json").read_text()))

pattern = np.array([[1,-1,1,-1]],dtype=float)
state,trace = binary_recall(store_binary(pattern),[1,-1,-1,-1])
assert np.array_equal(state,pattern[0])
assert trace[3]["field"] == .75 and trace[3]["energy"]-trace[2]["energy"] == -1.5
continuous = [float(np.tanh(-.3)),float(np.tanh(np.tanh(-.3)))]
gap = 10*np.exp(-3)
gap_results = {"competitor_factor":float(gap),"target_lower_bound":float(1/(1+gap)),
               "distance_upper_bound":float(2*gap/(1+gap))}
logits = torch.tensor([.3,-.2,.7],dtype=torch.float64,requires_grad=True)
probabilities = logits.softmax(dim=0)
mask = torch.tensor([True,False,True])
loss = -probabilities[mask].sum().log()
loss.backward()
conditional = torch.where(mask,probabilities/probabilities[mask].sum(),0.)
gradient_error = float((logits.grad-(probabilities-conditional)).abs().max().detach())
assert gradient_error < 1e-12

source = REPO/"src/learn/data/topics/modern-hopfield-networks.jsx"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == "d17566f43c90d83fbb24f0bb9ba51f6779a930a6b6cac366279a00d68fbdbcf5"
manifest = json.loads((REPO/"src/learn/data/lesson-manifest.json").read_text())
routes = sorted(set(re.findall(r"/learn/path/full-curriculum/([^?)]+)\?module=deep-learning-fundamentals",lesson)))
assert all(route in manifest for route in routes)
links = re.findall(r"\]\(\./([^)]*)\)",lesson)
assert all((ROOT/name).is_file() for name in links)
assert len(set(re.findall(r"\[Figure (\d+):",lesson)))==21
assert lesson.count("<details>")==20 and "<details open" not in lesson
assert len(re.findall(r"^### Investigation [A-D]",spec,re.M))==4
output = {"date":"2026-09-13","phase":"content author checks only",
          "source_sha256":source_hash,"source_lines":len(source.read_text(encoding="utf8").splitlines()),
          "inline_code_parity_maximum_errors":errors,"unique_feature_rows":5620,
          "disjoint_training_roles":{name:len(ids) for name,ids in role_ids.items()},
          "class_counts_balanced":True,"archive_finite":True,"confusion_counts_reconciled":True,
          "exercise_binary_change":trace[3],"exercise_continuous_reads":continuous,
          "exercise_gap":gap_results,"class_mass_gradient_maximum_error":gradient_error,
          "figure_count":21,"investigation_count":4,"closed_practice_disclosures":20,
          "verified_local_route_ids":routes,"verified_relative_downloads":links,
          "memory_bytes":256000000,"memory_MiB":256000000/2**20,
          "execution":"Executed current inline code and saved-model reads; no re-fit or browser review."}
(ROOT/"author-checks.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")
print(json.dumps(output,indent=2))
