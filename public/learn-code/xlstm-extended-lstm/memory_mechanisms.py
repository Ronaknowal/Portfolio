"""Exact, bounded xLSTM teaching operators. Run with NumPy and PyTorch installed."""
from pathlib import Path
import json
import numpy as np
import torch

ROOT=Path(__file__).parent


def scalar_scan(values,write_logs,forget_logs,outputs,stabilized=True):
    cell=normalizer=log_scale=0.
    trace=[]
    for value,write_log,forget_log,output in zip(values,write_logs,forget_logs,outputs):
        new_scale=max(forget_log+log_scale,write_log) if stabilized else 0.
        write=np.exp(write_log-new_scale)
        retain=np.exp(forget_log+log_scale-new_scale)
        cell=retain*cell+write*value
        normalizer=retain*normalizer+write
        hidden=output*cell/normalizer
        trace.append({"cell":float(cell),"normalizer":float(normalizer),"log_scale":float(new_scale),
                      "write":float(write),"retain":float(retain),"hidden":float(hidden)})
        log_scale=new_scale
    return trace


def matrix_scan(queries,keys,values,write_logs,forget_logs,stabilized=True,wrong_floor=False):
    cell=np.zeros((keys.shape[1],values.shape[1]))
    normalizer=np.zeros(keys.shape[1]);log_scale=0.
    trace=[]
    for query,key,value,write_log,forget_log in zip(queries,keys,values,write_logs,forget_logs):
        new_scale=max(forget_log+log_scale,write_log) if stabilized else 0.
        write=np.exp(write_log-new_scale);retain=np.exp(forget_log+log_scale-new_scale)
        cell=retain*cell+write*np.outer(key,value)
        normalizer=retain*normalizer+write*key
        numerator=cell.T@query
        floor=1. if wrong_floor else np.exp(-new_scale)
        denominator=max(abs(normalizer@query),floor)
        trace.append({"cell":cell.tolist(),"normalizer":normalizer.tolist(),"log_scale":float(new_scale),
                      "numerator":numerator.tolist(),"denominator":float(denominator),
                      "read":(numerator/denominator).tolist()})
        log_scale=new_scale
    return trace


def parallel_read(queries,keys,values,write_logs,forget_logs):
    length=len(queries)
    cumulative=np.cumsum(forget_logs)
    log_gate=write_logs[None,:]+cumulative[:,None]-cumulative[None,:]
    mask=np.arange(length)[:,None]>=np.arange(length)[None,:]
    log_gate=np.where(mask,log_gate,-np.inf)
    scale=log_gate.max(axis=1,keepdims=True)
    gate=np.exp(log_gate-scale)
    coefficients=(queries@keys.T)*gate
    numerator=coefficients@values
    denominator=np.maximum(np.abs(coefficients.sum(axis=1,keepdims=True)),np.exp(-scale))
    return numerator/denominator


def chunk_read(queries,keys,values,write_logs,forget_logs,chunk_size,initial=None):
    # Unscaled form for moderate inputs: explicit incoming state plus local causal work.
    cell=np.zeros((keys.shape[1],values.shape[1])) if initial is None else initial[0].copy()
    normalizer=np.zeros(keys.shape[1]) if initial is None else initial[1].copy()
    output=[]
    for start in range(0,len(queries),chunk_size):
        end=min(start+chunk_size,len(queries))
        q,k,v=queries[start:end],keys[start:end],values[start:end]
        cumulative=np.cumsum(forget_logs[start:end])
        local_log=write_logs[start:end][None,:]+cumulative[:,None]-cumulative[None,:]
        mask=np.arange(end-start)[:,None]>=np.arange(end-start)[None,:]
        local_gate=np.exp(np.where(mask,local_log,-np.inf))
        coefficients=(q@k.T)*local_gate
        incoming=np.exp(cumulative)
        numerator=(q@cell)*incoming[:,None]+coefficients@v
        mass=(q@normalizer)*incoming+coefficients.sum(axis=1)
        output.extend(numerator/np.maximum(np.abs(mass[:,None]),1.))
        end_weights=np.exp(write_logs[start:end]+cumulative[-1]-cumulative)
        cell=incoming[-1]*cell+k.T@(end_weights[:,None]*v)
        normalizer=incoming[-1]*normalizer+k.T@end_weights
    return np.array(output),(cell,normalizer)


def main():
    values=np.array([.2,-.6,.8]);write=np.log([1.,3.,9.]);forget=np.log([.5,.5,.5]);outputs=np.full(3,.75)
    raw=scalar_scan(values,write,forget,outputs,False)
    stable=scalar_scan(values,write,forget,outputs)
    fresh_values=np.array([-.4,.7,-.2]);fresh_write=np.log([2.,1.,5.]);fresh_forget=np.log([.8,.6,.4])
    fresh=scalar_scan(fresh_values,fresh_write,fresh_forget,np.full(3,.8))
    changed=scalar_scan(fresh_values,np.log([2.,1.,.5]),fresh_forget,np.full(3,.8))
    scalar_error=max(abs(a["hidden"]-b["hidden"]) for a,b in zip(raw,stable))
    shifted=scalar_scan(values,write+1000,forget,outputs)
    shift_error=max(abs(a["hidden"]-b["hidden"]) for a,b in zip(stable,shifted))
    constant=scalar_scan(np.full(3,.6),fresh_write,fresh_forget,np.full(3,.8))
    q=np.array([[1.,0.],[0.,1.],[1.,0.]])
    k=q.copy();v=np.array([[2.,-1.],[0.,3.],[4.,1.]])
    ilog=np.log([1.,1.,2.]);flog=np.log([.5,.5,.5])
    matrix=matrix_scan(q,k,v,ilog,flog,False)
    matrix_stable=matrix_scan(q,k,v,ilog,flog)
    cancellation=matrix_scan(np.array([[1.,0.],[1.,0.]]),np.array([[1.,0.],[-1.,0.]]),
                             np.array([[2.],[-1.]]),np.zeros(2),np.zeros(2),False)
    bug_args=(np.array([[.5]]),np.array([[.25]]),np.array([[4.]]),np.array([2.]),np.array([0.]))
    floor_case={"raw":matrix_scan(*bug_args,False),"stable":matrix_scan(*bug_args),
                "wrong_floor":matrix_scan(*bug_args,wrong_floor=True)}
    random=np.random.default_rng(229)
    inputs=(random.normal(size=(7,3)),random.normal(size=(7,3)),random.normal(size=(7,2)),
            random.uniform(-.7,.9,size=7),random.uniform(-1.2,-.05,size=7))
    recurrent=np.array([row["read"] for row in matrix_scan(*inputs)])
    dense=parallel_read(*inputs)
    chunk_errors={str(size):float(np.max(np.abs(chunk_read(*inputs,size)[0]-recurrent)))
                  for size in [1,2,3,4,7,9]}
    initial=(random.normal(size=(3,2)),random.normal(size=3))
    initial_reference=chunk_read(*inputs,1,initial)[0]
    carry_errors={str(size):float(np.max(np.abs(chunk_read(*inputs,size,initial)[0]-initial_reference)))
                  for size in [2,3,7,9]}
    future=list(inputs);future[2]=future[2].copy();future[2][4:]+=7
    causal_error=float(np.max(np.abs(parallel_read(*future)[:4]-dense[:4])))
    # A cell-level differentiable check: stabilize the scalar recurrence without changing its gradient.
    base=torch.tensor([.3,-.4,.8],dtype=torch.float64)
    gradients=[];losses=[]
    for stabilized in [False,True]:
        logits=base.clone().requires_grad_();cell=normalizer=scale=0.
        for index in range(3):
            new_scale=torch.maximum(logits[index],torch.as_tensor(scale-.2,dtype=torch.float64)) if stabilized else 0.
            i=(logits[index]-new_scale).exp();f=torch.as_tensor(-.2+scale-new_scale,dtype=torch.float64).exp()
            cell=f*cell+i*values[index];normalizer=f*normalizer+i;scale=new_scale
        loss=(cell/normalizer-.4)**2
        loss.backward();gradients.append(logits.grad.tolist());losses.append(float(loss.detach()))
    gradient_error=float(np.max(np.abs(np.array(gradients[0])-np.array(gradients[1]))))
    results={"scalar_worked_raw":raw,"scalar_worked_stable":stable,"scalar_fresh":fresh,
             "scalar_fresh_weaker_last_write":changed,"scalar_constant_content_null":constant,
             "scalar_raw_stable_error":scalar_error,"scalar_log1000_shift_error":shift_error,
             "matrix_worked_raw":matrix,"matrix_worked_stable":matrix_stable,
             "signed_cancellation":cancellation,"scaled_floor_case":floor_case,
             "dense_recurrent_error":float(np.max(np.abs(dense-recurrent))),
             "chunk_errors":chunk_errors,"nonzero_carry_errors":carry_errors,
             "future_edit_prefix_error":causal_error,"gradient_losses":losses,
             "gradient_vectors":gradients,"gradient_maximum_error":gradient_error,
             "seven_b_state":{"blocks":32,"heads":8,"key_width":256,"value_width":512,
                 "matrix_bytes":32*8*256*512*4,"matrix_normalizer_scale_bytes":32*8*(256*512+256+1)*4}}
    assert scalar_error<1e-12 and shift_error<1e-12 and gradient_error<1e-12
    assert max(chunk_errors.values())<1e-12 and max(carry_errors.values())<1e-12
    assert causal_error==0.
    (ROOT/"mechanism-results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
