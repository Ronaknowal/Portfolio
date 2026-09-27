"""Reproduce fresh investigation fixtures from retained fits; does not train."""
from pathlib import Path
import json
import numpy as np
import torch
from memory_mechanisms import scalar_scan, matrix_scan, chunk_read
from row_sequence_models import DigitReader, load_data

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(1)
    theta = 0.
    def prediction(parameter):
        weight = np.exp(parameter)
        return (-.1 + .8*weight)/(.5 + weight)
    before = prediction(theta)
    derivative = .5/(1.5**2)
    gradient = (before-.7)*derivative
    after_theta = theta-.5*gradient
    gate_step = {"theta_before": theta, "prediction_before": before, "gradient": gradient,
                 "theta_after": after_theta, "prediction_after": prediction(after_theta),
                 "loss_before": .5*(before-.7)**2, "loss_after": .5*(prediction(after_theta)-.7)**2}
    keys = np.array([[1.,0.],[.5,1.],[-.5,1.]])
    values = np.array([[1.,2.],[-2.,1.],[3.,-1.]])
    queries = np.array([[1.,0.],[0.,1.],[.5,1.]])
    write_logs=np.log([1.,2.,1.]);forget_logs=np.log([.8,.5,.7])
    fresh=matrix_scan(queries,keys,values,write_logs,forget_logs,False)
    changed_keys=keys.copy();changed_keys[-1]=[.5,-1.]
    changed=matrix_scan(queries,changed_keys,values,write_logs,forget_logs,False)
    zero=matrix_scan(queries,keys,np.zeros_like(values),write_logs,forget_logs,False)
    zero_query=matrix_scan(np.zeros_like(queries),keys,values,write_logs,forget_logs,False)
    random=np.random.default_rng(229)
    chunk_inputs=(random.normal(size=(7,3)),random.normal(size=(7,3)),random.normal(size=(7,2)),
                  random.uniform(-.7,.9,size=7),random.uniform(-1.2,-.05,size=7))
    incoming=(random.normal(size=(3,2)),random.normal(size=3))
    whole=chunk_read(*chunk_inputs,3)[0]
    reset_chunks=np.concatenate([chunk_read(*(value[start:start+3] for value in chunk_inputs),3)[0]
                          for start in range(0,7,3)])
    future_inputs=list(chunk_inputs);future_inputs[2]=future_inputs[2].copy();future_inputs[2][4:]+=7
    future_outputs=chunk_read(*future_inputs,3)[0]
    assert np.max(np.abs(whole[:3]-reset_chunks[:3]))<1e-12
    assert np.max(np.abs(whole[:4]-future_outputs[:4]))<1e-12
    assert np.max(np.abs(whole[3:]-reset_chunks[3:]))>1e-3
    scalar_exercise=scalar_scan(np.array([-.5,.25,1.]),np.log([2.,1.,4.]),
        np.log([.5,.5,.25]),np.ones(3),False)
    arrays=np.load(ROOT/"row-sequence-fits.npz")
    roles,metadata=load_data()
    images,labels=roles["validation"]
    fixtures=[]
    for index in [35,142]:
        image=images[index:index+1]
        conditions={"clean":image,"reversed":image.flip(1),"bottom_zero":image.clone(),
                    "blank":torch.zeros_like(image)}
        conditions["bottom_zero"][:,5:]=0.
        record={"validation_index":index,"training_source_id":metadata["training_source_ids"]["validation"][index],
                "label":int(labels[index]),"pixels":(image[0]*16).int().tolist(),"models":{}}
        for kind in ["lstm","slstm","mlstm"]:
            model=DigitReader(kind)
            prefix=f"{kind}_seed19__"
            state={key[len(prefix):]:torch.from_numpy(arrays[key].copy()) for key in arrays.files if key.startswith(prefix)}
            model.load_state_dict(state);model.eval()
            outputs={}
            with torch.no_grad():
                for name,inputs in conditions.items():
                    logits,final_state=model(inputs)
                    outputs[name]={"logits":logits[0].tolist(),"predictions":logits[0].argmax(-1).tolist(),
                                   "probabilities":logits[0].softmax(-1).tolist()}
                first,left_state=model(image[:,:3]);continued,_=model(image[:,3:],left_state)
                reset,_=model(image[:,3:])
                full=torch.tensor(outputs["clean"]["logits"])
                error=float((full-torch.cat([first,continued],dim=1)[0]).abs().max())
                prefix_error=float((full[:5]-torch.tensor(outputs["bottom_zero"]["logits"])[:5]).abs().max())
                state_trace=[];carried=None
                for row in range(8):
                    _,carried=model(image[:,row:row+1],carried)
                    state_trace.append([entry[0].tolist() if kind!="lstm" else entry[:,0].tolist() for entry in carried])
                outputs["carry_maximum_error"]=error
                outputs["future_edit_prefix_error"]=prefix_error
                outputs["reset_final_logits"]=reset[0,-1].tolist()
                outputs["state_trace"]=state_trace
                assert error<1e-5 and prefix_error<1e-5
            record["models"][kind]=outputs
        fixtures.append(record)
    independent_scalar_inputs={"values":[.1,-.8,.5,.3],"writes":[1.,4.,2.,6.],"retention":[.9,.7,.4,.8],"output":.6}
    independent_scalar=scalar_scan(np.array(independent_scalar_inputs["values"]),np.log(independent_scalar_inputs["writes"]),
        np.log(independent_scalar_inputs["retention"]),np.full(4,.6))
    independent_changed=scalar_scan(np.array(independent_scalar_inputs["values"]),np.log([1.,4.,2.,.75]),
        np.log(independent_scalar_inputs["retention"]),np.full(4,.6))
    results={"scalar_independent_inputs":independent_scalar_inputs,"scalar_independent":independent_scalar,
        "scalar_independent_weaker_last_write":independent_changed,
        "gate_gradient_step":gate_step,"matrix_fresh_inputs":{"keys":keys.tolist(),"values":values.tolist(),
        "queries":queries.tolist(),"write":[1,2,1],"forget":[.8,.5,.7]},"matrix_fresh":fresh,
        "matrix_changed_last_key":changed,"matrix_zero_values_null":zero,"matrix_zero_query_null":zero_query,
        "chunk_inputs":{name:value.tolist() for name,value in zip(["queries","keys","values","write_logs","forget_logs"],chunk_inputs)},
        "chunk_incoming_state":{"cell":incoming[0].tolist(),"normalizer":incoming[1].tolist()},
        "chunk_outputs":whole.tolist(),"chunk_reset_every_three":reset_chunks.tolist(),
        "chunk_reset_maximum_difference":float(np.max(np.abs(whole-reset_chunks))),
        "chunk_future_edit_outputs":future_outputs.tolist(),
        "scalar_practice":scalar_exercise,"digit_fixtures":fixtures}
    (ROOT/"investigation-results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"gate_step":gate_step,"matrix_fresh_final":fresh[-1],"matrix_changed_final":changed[-1],
        "practice_final":scalar_exercise[-1],"digits":[{"source":x["training_source_id"],"label":x["label"],
        "predictions":{kind:{condition:result[condition]["predictions"] for condition in ["clean","reversed","bottom_zero","blank"]}
        for kind,result in x["models"].items()}} for x in fixtures]},indent=2))


if __name__ == "__main__":
    main()
