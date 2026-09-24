"""Bounded author calculations for narrative/visual fixtures; no UI implementation."""
from pathlib import Path
import json
import numpy as np
import torch
from trajectory_state_models import prepare, TrajectoryClassifier
from state_space_mechanisms import selective_scalar, held_discretize, recurrent, kernel, fft_convolve, ssd_recurrent, decay_matrix, ssd_chunked

def main():
    torch.set_num_threads(2)
    folder=Path(__file__).parent
    x,labels,roles,_=prepare(folder)
    saved=np.load(folder/"trajectory-state-fits.npz")
    record={"gradient_example":{"state":[1.,2.],"target":3.,"C_before":[.5,.5],
             "output_before":1.5,"loss_before":1.125,"gradient":[-1.5,-3.],
             "step":.1,"C_after":[.65,.8],"output_after":2.25,"loss_after":.28125},
            "real_input_fixtures":[]}
    C=torch.tensor([.5,.5],requires_grad=True);state=torch.tensor([1.,2.])
    loss=.5*(C@state-3).square();loss.backward()
    assert torch.allclose(C.grad,torch.tensor([-1.5,-3.]))
    # Predetermined first validation row, not selected on test outcomes.
    row=int(roles[1][0]); original=x[row:row+1].clone()
    record["source_row_id"]=row+1;record["true_class"]=int(labels[row])+1
    for kind in ["diagonal","selective"]:
        model=TrajectoryClassifier(kind)
        prefix=f"{kind}_seed17::"
        model.load_state_dict({k:torch.from_numpy(saved[prefix+k]) for k in model.state_dict()})
        model.eval()
        edited=original.clone();edited[0,22,0]*=-1
        with torch.no_grad():
            full=model(original);edit=model(edited);reverse=model(original.flip(1))
            serial_logits=model(original,recurrent=True)
            reset=model(original) # new call initializes all state.
        def describe(v):
            return {"prediction":int(v.argmax(1))+1,"true_class_probability":float(v.softmax(1)[0,labels[row]]),
                    "logits":v[0].tolist()}
        record["real_input_fixtures"].append({"kind":kind,"seed":17,"point23_x_before":float(original[0,22,0]),
            "point23_x_after":float(edited[0,22,0]),"base":describe(full),"edited":describe(edit),
            "reversed_order":describe(reverse),"reset_max_difference":float((reset-full).abs().max()),
            "alternate_serial_max_difference":float((serial_logits-full).abs().max())})
        # Causality is tested at temporal mixer output, since final mean pools all times.
        with torch.no_grad():
            u=model.norms[0](model.input(original));changed=u.clone();changed[:,30:]=2
            before=model.mixers[0](u);after=model.mixers[0](changed)
            delta=float((before[:,:30]-after[:,:30]).abs().max())
        assert delta<2e-5
        record["real_input_fixtures"][-1]["future_edit_prefix_max_difference"]=delta

    Ad,Bd=held_discretize(np.diag([-1.,-2.]),np.ones((2,1)),np.log(2))
    Cread=np.array([1.,-.5]); signal=np.array([1.,-2.,3.,0.])
    output,_=recurrent(Ad,Bd,Cread,.25,signal)
    taps=kernel(Ad,Bd,Cread,4)
    assert np.allclose(output,np.convolve(signal,taps)[:4]+.25*signal)
    assert np.allclose(output,fft_convolve(signal,taps)+.25*signal)
    marked_signal=[3.,-8.,5.,-2.]; gates=[.99,.01,.01,.99]
    selected=selective_scalar(marked_signal,gates); constant=selective_scalar(marked_signal,[.5]*4)
    a=np.array([.5,.5,.25,.8]); b=np.array([[1,0],[0,1],[1,1],[1,-1.]])
    c=np.array([[1,0],[1,1],[0,1],[1,2.]])
    v=np.array([[2,1],[3,-1],[-1,3],[-2,1.]])
    ssd,_=ssd_recurrent(a,b,c,v)
    assert np.allclose(ssd,((c@b.T)*decay_matrix(a))@v)
    for q in [1,2,3,4,8]:
        assert np.allclose(ssd,ssd_chunked(a,b,c,v,q)[0])

    # Check the new optional finite-kernel and rank-one resolvent derivations.
    z=np.exp(2j*np.pi/7); identity=np.eye(2)
    finite=float(0)+sum(taps[l]*z**l for l in range(4))
    rational=Cread@(identity-np.linalg.matrix_power(z*Ad,4))@np.linalg.solve(identity-z*Ad,Bd[:,0])
    assert abs(finite-rational)<1e-12
    lam=np.array([-.5+1j,-.5-1j]); pv=np.array([.3+.1j,.3-.1j]); qv=np.array([.2-.4j,.2+.4j])
    at=np.diag(lam)-np.outer(pv,qv.conj()); point=1+2j
    r0=np.diag(1/(point-lam))
    wood=r0-np.outer(r0@pv,qv.conj()@r0)/(1+qv.conj()@r0@pv)
    direct=np.linalg.inv(point*identity-at)
    assert np.allclose(wood,direct,atol=1e-12)
    record["advanced_identity_checks"]={"finite_kernel_max_error":float(abs(finite-rational)),
        "woodbury_max_error":float(abs(wood-direct).max()),"rank_two_write":int(np.linalg.matrix_rank(np.eye(2))),
        "rank_one_write":int(np.linalg.matrix_rank(np.outer([1.,0],[1.,0]))),
        "conjugate_pair_real_output":float(2*((1+2j)*(3-1j)).real)}

    record["unsolved_lab_defaults"]={"system":{"inputs":signal.tolist(),"outputs":output.tolist()},
        "selection":{"inputs":marked_signal,"selective":selected,"constant":constant,
                     "read_index2_target":3.,"errors":[abs(selected[2]-3),abs(constant[2]-3)]},
        "ssd":{"values":v.tolist(),"outputs":ssd.tolist(),"chunk_size":3},
        "closed_write_independent_decay":{"initial":4.,"a":.5,"g":0.,"states":[2.,1.,.5,.25]}}

    (folder/"investigation-checks.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in record.items() if k!="real_input_fixtures"}))
    for r in record["real_input_fixtures"]:
        print(r["kind"],{k:r[k] for k in ["point23_x_before","point23_x_after","future_edit_prefix_max_difference"]},
            {k:(r[k]["prediction"],r[k]["true_class_probability"]) for k in ["base","edited","reversed_order"]})

if __name__=="__main__":
    main()
