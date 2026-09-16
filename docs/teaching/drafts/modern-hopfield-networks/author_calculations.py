"""Finite author fixtures for the pending visual investigations; no UI implementation."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from digit_memory import load_roles, read_memory, obscure
from associative_memory import softmax, energy, retrieve

ROOT = Path(__file__).parent
torch.set_num_threads(1)
roles, metadata = load_roles()
archive = np.load(ROOT/"digit-memory-fits.npz")
memory, memory_labels = roles["memory"]
validation, validation_labels = roles["validation"]
projection = nn.Linear(64,16,bias=False)
projection.weight.data.copy_(torch.from_numpy(archive["seed17_projection"]))
results = {"digit_fixtures":{},"association_fixtures":{}}
with torch.no_grad():
    for name, index in [("worked",30),("fresh",31),("successful_zero",0)]:
        clean = validation[index:index+1]
        edited = obscure(clean)
        # A genuinely editable observed pixel, independent of the provided occlusion button.
        pixel_edit = clean.clone()
        pixel_edit[0,28] = 1-pixel_edit[0,28]
        fixtures = {}
        for condition, cue in [("clean",clean),("central_columns_zero",edited),
                                ("pixel_r4_c5_complement",pixel_edit),
                                ("restore_null",clean.clone()),("blank_null",torch.zeros_like(clean))]:
            log_class, weights = read_memory(cue,memory,memory_labels,16.,projection)
            top = torch.argsort(weights[0],descending=True)[:3]
            image = weights@memory
            fixtures[condition] = {"predicted_class":int(log_class.argmax()),
                 "true_class_mass":float(log_class[0,validation_labels[index]].exp()),
                 "class_mass":log_class[0].exp().tolist(),
                 "maximum_memory_weight":float(weights.max()),
                 "top_memory_source_ids":[metadata["roles_training_source_ids"]["memory"][int(i)] for i in top],
                 "top_weights":weights[0,top].tolist(),
                 "read_image_mse_to_clean":float(((image-clean)**2).mean()),
                 "input_mse_to_clean":float(((cue-clean)**2).mean()),
                 "read_image":image[0].tolist()}
        clean_log, clean_weights = read_memory(clean,memory,memory_labels,16.,projection)
        restored_log, restored_weights = read_memory(clean.clone(),memory,memory_labels,16.,projection)
        fixtures["restore_maximum_error"] = float((clean_weights-restored_weights).abs().max())
        results["digit_fixtures"][name] = {"validation_index_zero_based":index,
             "source_id":metadata["roles_training_source_ids"]["validation"][index],
             "true_class":int(validation_labels[index]),"pixel_r4_c5_original":float(clean[0,28]),
             "conditions":fixtures}
    # Same nonzero normalization scale in cue and bank leaves all scores unchanged up to rounding.
    original,_ = read_memory(validation[:5],memory,memory_labels,16.,projection)
    scaled,_ = read_memory(2*validation[:5],2*memory,memory_labels,16.,projection)
    results["positive_scale_null_maximum_log_class_error"] = float((original-scaled).abs().max())
    gram = (memory@projection.weight.T)
    gram = torch.nn.functional.normalize(gram,dim=-1)
    norm_error = (gram.norm(dim=-1)-1).abs().max()
    results["memory_key_norm_maximum_error"] = float(norm_error)

# Association workspace: key-space geometry and value-space payload remain distinct.
for name,keys,values,query,beta in [
    ("worked",[[1,0],[0,1]],[[1,0],[0,1]],[.6,-.2],1.),
    ("fresh",[[1,0],[0,1],[-1,0]],[[1,0],[0,1],[0,1]],[-.3,.4],2.),
    ("fresh_query_changed",[[1,0],[0,1],[-1,0]],[[1,0],[0,1],[0,1]],[.8,-.4],2.),
    ("fresh_payload_changed",[[1,0],[0,1],[-1,0]],[[1,0],[0,1],[1,0]],[-.3,.4],2.),
    ("fresh_values_equal_null",[[1,0],[0,1],[-1,0]],[[4,-2],[4,-2],[4,-2]],[-.3,.4],2.),
]:
    keys,values,query=np.array(keys,float),np.array(values,float),np.array(query,float)
    weights=softmax(beta*keys@query)
    results["association_fixtures"][name]={"scores":(keys@query).tolist(),"weights":weights.tolist(),
         "key_read":(weights@keys).tolist(),"value_read":(weights@values).tolist()}

# Null mechanisms and error/derivative checks independent of a preset answer.
patterns=np.array([[1.,0.],[-1.,0.]])
cue=np.array([.2,.4]); epsilon=1e-6
analytic=cue-retrieve(patterns,cue,2.)[0]
numeric=np.array([(energy(patterns,cue+epsilon*np.eye(2)[i],2.)-
                   energy(patterns,cue-epsilon*np.eye(2)[i],2.))/(2*epsilon) for i in range(2)])
results["energy_gradient_maximum_error"]=float(np.max(np.abs(analytic-numeric)))
results["synchronous_two_cycle"]=[[1,-1],[-1,1],[1,-1]]
results["capacity_memory_bytes"]={"patterns":1_000_000,"dimension":64,"dtype_bytes":4,
                                   "stored_pattern_bytes":1_000_000*64*4}
results["mask_training_labels_only"]={"memory":200,"query_fit":800,"validation":300,"test":1797}
assert results["energy_gradient_maximum_error"] < 1e-8
assert results["digit_fixtures"]["fresh"]["conditions"]["restore_maximum_error"] == 0.
(ROOT/"investigation-checks.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps({"digits":{name:{"source":f["source_id"],"truth":f["true_class"],
      "conditions":{condition:{k:v for k,v in result.items() if k not in ["class_mass","read_image"]}
           for condition,result in f["conditions"].items() if isinstance(result,dict)}}
      for name,f in results["digit_fixtures"].items()},"association":results["association_fixtures"],
      "gradient_error":results["energy_gradient_maximum_error"]},indent=2))
