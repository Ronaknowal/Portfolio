"""Bounded author closure: manuscript examples, retained fits and source-bound inputs."""
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import io
import json
import re
import subprocess
import numpy as np
import torch
from row_sequence_models import DigitReader, load_data

ROOT=Path(__file__).parent
REPOSITORY=ROOT.parents[3]


def main():
    torch.set_num_threads(1)
    lesson=(ROOT/"lesson.md").read_text(encoding="utf8")
    specifications=(ROOT/"visual-specifications.md").read_text(encoding="utf8")
    programs=re.findall(r"```python\n(.*?)```",lesson,re.S)
    assert len(programs)==1
    captured=io.StringIO()
    with redirect_stdout(captured): exec(compile(programs[0],"lesson-inline","exec"),{})
    assert captured.getvalue()=="0.150000\n-0.364286\n0.443023\n"
    reports=json.loads((ROOT/"row-sequence-results.json").read_text())
    mechanisms=json.loads((ROOT/"mechanism-results.json").read_text())
    fixtures=json.loads((ROOT/"investigation-results.json").read_text())
    roles,metadata=load_data()
    assert metadata==reports["data"]
    fit=set(metadata["training_source_ids"]["fit"])
    validation=set(metadata["training_source_ids"]["validation"])
    assert len(fit)==1000 and len(validation)==300 and not fit.intersection(validation)
    arrays=np.load(ROOT/"row-sequence-fits.npz")
    checks=[]
    for report in reports["models"]:
        kind,seed=report["kind"],report["seed"]
        model=DigitReader(kind)
        prefix=f"{kind}_seed{seed}__"
        state={key[len(prefix):]:torch.from_numpy(arrays[key].copy()) for key in arrays.files if key.startswith(prefix)}
        assert all(torch.isfinite(value).all() for value in state.values())
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            logits,_=model(roles["validation"][0])
        error=float(np.max(np.abs(logits.numpy()-arrays[f"{kind}_seed{seed}_clean_logits"])))
        assert error<1e-6
        assert sum(parameter.numel() for parameter in model.parameters())==report["parameters"]
        for role in ["fit","validation","test"]:
            for condition in ["clean","reversed"]:
                metric=report[f"{role}_{condition}"]
                confusion=np.array(metric["confusion"])
                assert confusion.sum()==metric["count"]
                assert confusion.sum()-confusion.trace()==metric["errors"]
        checks.append({"model":kind,"seed":seed,"saved_logit_maximum_error":error,
                       "parameters":report["parameters"],"confusions_reconciled":True})
    assert mechanisms["scaled_floor_case"]["raw"][0]["read"][0]>3.69
    assert mechanisms["scaled_floor_case"]["wrong_floor"][0]["read"]==[.5]
    assert mechanisms["gradient_maximum_error"]<1e-12
    assert max(mechanisms["chunk_errors"].values())<1e-12
    assert fixtures["scalar_independent"][-1]["hidden"]>fixtures["scalar_independent_weaker_last_write"][-1]["hidden"]
    assert all(np.allclose(row["read"],0.) for row in fixtures["matrix_zero_values_null"])
    assert all(np.allclose(row["read"],0.) for row in fixtures["matrix_zero_query_null"])
    assert np.allclose(np.array(fixtures["chunk_outputs"])[:3],np.array(fixtures["chunk_reset_every_three"])[:3])
    assert fixtures["chunk_reset_maximum_difference"]>5.
    assert 12*4*(32*64+32+1)*4==399552
    assert abs((5*.2*3*.5)/max(abs(5*.2*.5),1)-1.5)<1e-12
    routes=re.findall(r"/learn/path/full-curriculum/([^?]+)\?module=deep-learning-fundamentals",lesson)
    node="import { topicCatalogue } from './src/learn/data/curriculum/topic-catalogue.js'; const ids="+json.dumps(routes)+"; console.log(JSON.stringify(ids.map(id=>({id,tracks:topicCatalogue[id]?.trackIds}))));"
    route_data=json.loads(subprocess.run(["node","--input-type=module","-e",node],cwd=REPOSITORY,
        capture_output=True,text=True,check=True).stdout)
    assert len(routes)==4 and all("deep-learning-fundamentals" in row["tracks"] for row in route_data)
    figures=re.findall(r"\[Figure (X\d\d):",lesson)
    investigations=re.findall(r"\[Investigation (X[A-D]):",lesson)
    assert len(figures)==len(set(figures))==20 and len(investigations)==4
    assert all(identifier in specifications for identifier in figures+investigations)
    assert lesson.count("<details>")==lesson.count("</details>")==20 and "<details open" not in lesson
    original=REPOSITORY/"src/learn/data/topics/xlstm-extended-lstm.jsx"
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    assert original_hash=="d9594e806e11b12e8eedbd763c141da9b36bf9067a4ef82e986e80ee80f9b995"
    report={"date":"2026-09-13","phase":"author content checks only","inline_program_output":captured.getvalue(),
        "data_hashes":metadata["hashes"],"feature_rows_unique":metadata["unique_feature_rows"],
        "fit_rows":1000,"validation_rows":300,"test_rows":1797,"role_overlap":0,
        "saved_models":checks,"routes":route_data,"figures":figures,"investigations":investigations,
        "closed_practice_details":20,"original_source_sha256":original_hash,
        "known_mathematical_fixtures_checked":True,"implementation_or_browser_checks":False}
    (ROOT/"author-checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
