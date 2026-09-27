"""Executed author calculations; no retraining. All synthetic cases labelled."""
from pathlib import Path
import json
import math
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import torch
from neural_ode_study import DepthClassifier, integrate, load_data

DIRECTORY = Path(__file__).resolve().parent


def fixed_solve(field, initial, endpoint, steps, method="rk4"):
    state = np.asarray(initial, dtype=float).copy()
    trace = [state.copy()]
    step_size = endpoint/steps
    for index in range(steps):
        time = index*step_size
        first = field(time, state)
        if method == "euler":
            state = state + step_size*first
        else:
            second = field(time+step_size/2, state+step_size*first/2)
            third = field(time+step_size/2, state+step_size*second/2)
            fourth = field(time+step_size, state+step_size*third)
            state = state + step_size*(first+2*second+2*third+fourth)/6
        trace.append(state.copy())
    return np.stack(trace)


def adaptive_heun(field, initial, endpoint, tolerance, initial_step=.1):
    """Embedded Euler/Heun, RMS scaled error, with every attempted step retained."""
    state, time, step_size = np.asarray(initial, dtype=float), 0., initial_step
    attempts = []
    while time < endpoint:
        step_size = min(step_size, endpoint-time)
        first = field(time, state)
        euler = state + step_size*first
        second = field(time+step_size, euler)
        heun = state + step_size*(first+second)/2
        scale = tolerance*.01 + tolerance*np.maximum(np.abs(state), np.abs(heun))
        ratio = float(np.sqrt(np.mean(((heun-euler)/scale)**2)))
        accepted = ratio <= 1.
        attempts.append(dict(time=time, step_size=step_size, before=state.tolist(),
            euler=euler.tolist(), heun=heun.tolist(), error_ratio=ratio, accepted=accepted))
        if accepted:
            state, time = heun, time+step_size
        factor = 5. if ratio==0. else min(5., max(.1, .9*ratio**(-.5)))
        step_size *= factor
        if len(attempts)>10000 or step_size<1e-14:
            raise RuntimeError("bounded demonstration exhausted")
    return dict(endpoint=state.tolist(), attempts=attempts, nfe=2*len(attempts),
                accepted=sum(row["accepted"] for row in attempts),
                rejected=sum(not row["accepted"] for row in attempts))


def scalar_gradient(rate, initial, target, endpoint, steps, method):
    parameter = torch.tensor(rate, requires_grad=True)
    prediction = integrate(lambda time,state: parameter*state,
        torch.tensor([[initial]]), steps, method, endpoint)[-1,0,0]
    loss = .5*(prediction-target)**2
    gradient = float(torch.autograd.grad(loss, parameter)[0])
    def objective(value):
        output = fixed_solve(lambda time,state:value*state,[initial],endpoint,steps,method)[-1,0]
        return .5*(output-target)**2
    finite_difference=(objective(rate+1e-6)-objective(rate-1e-6))/(2e-6)
    exact_output=initial*math.exp(rate*endpoint)
    exact_gradient=(exact_output-target)*endpoint*exact_output
    # Continuous augmented adjoint equations integrated backwards from numerical output.
    def augmented(time, state):
        value, adjoint, accumulator = state
        return np.array([rate*value, -rate*adjoint, -adjoint*value])
    terminal=np.array([float(prediction.detach()),float(prediction.detach())-target,0.])
    back=fixed_solve(augmented,terminal,-endpoint,steps,"rk4")[-1]
    assert abs(gradient-finite_difference)<2e-8
    return dict(rate=rate,initial=initial,target=target,endpoint=endpoint,steps=steps,
        method=method, prediction=float(prediction.detach()), gradient=gradient,
        central_difference=finite_difference, exact_prediction=exact_output,
        exact_continuous_gradient=exact_gradient, backwards_continuous_accumulator=float(back[2]),
        reconstructed_initial=float(back[0]))


def observations(times, values, query, omit=None):
    state,time=0.,0.
    events=[]
    for index,(observed_at,value) in enumerate(zip(times,values)):
        if index==omit or observed_at>query:
            continue
        before=state*math.exp(-.5*(observed_at-time))
        state=.7*before+.3*value
        events.append(dict(time=observed_at,value=value,before=before,after=state))
        time=observed_at
    return dict(events=events,query=query,state=state*math.exp(-.5*(query-time)))


def native_model(snapshot):
    model=DepthClassifier(snapshot["kind"])
    model.load_state_dict({key:torch.tensor(value) for key,value in snapshot["state"].items()})
    model.eval()
    return model


def independent_inference(snapshot, initial, steps=4, method="rk4"):
    weights={key:np.asarray(value) for key,value in snapshot["state"].items()}
    dimension=6 if snapshot["kind"]=="augmented_ode" else 4
    initial=np.concatenate((initial,np.zeros(dimension-4)))
    def field(time,state):
        values=np.concatenate((state,[time]))
        hidden=np.tanh(weights["field.input.weight"]@values+weights["field.input.bias"])
        return weights["field.output.weight"]@hidden+weights["field.output.bias"]
    trace=fixed_solve(field,initial,1.,steps,method)
    logits=weights["readout.weight"]@trace[-1]+weights["readout.bias"]
    probabilities=np.exp(logits-logits.max())
    probabilities/=probabilities.sum()
    return dict(trace=trace.tolist(),logits=logits.tolist(),probabilities=probabilities.tolist())


def main():
    results={}
    results["rotation"]=[]
    for initial,endpoint in [([1.,0.],1.),([.6,.8],1.2)]:
        matrix=np.array([[0.,-1.],[1.,0.]])
        exact=expm(matrix*endpoint)@initial
        for method in ["euler","rk4"]:
            for steps in [4,8,16,32]:
                trace=fixed_solve(lambda time,state:matrix@state,initial,endpoint,steps,method)
                results["rotation"].append(dict(initial=initial,endpoint=endpoint,method=method,
                    steps=steps,trace=trace.tolist(),exact=exact.tolist(),
                    error=float(np.linalg.norm(trace[-1]-exact)),norm=float(np.linalg.norm(trace[-1]))))
    results["adaptive"]=[]
    for rates,initial,endpoint in [([-1.,-100.],[1.,1.],.4),([-2.,-50.],[1.,.3],.4)]:
        for tolerance in [.01,.001]:
            result=adaptive_heun(lambda time,state:np.array(rates)*state,initial,endpoint,tolerance)
            result.update(rates=rates,initial=initial,endpoint_time=endpoint,tolerance=tolerance,
                exact=(np.array(initial)*np.exp(np.array(rates)*endpoint)).tolist())
            results["adaptive"].append(result)
    results["stiff_slow_manifold"]=[]
    for stiffness in [5.,1000.]:
        for method in ["RK45","Radau"]:
            field=lambda time,state:-stiffness*(state-np.cos(time))-np.sin(time)
            solution=solve_ivp(field,(0.,1.),[1.],method=method,rtol=1e-6,atol=1e-9)
            results["stiff_slow_manifold"].append(dict(stiffness=stiffness,method=method,
                nfe=int(solution.nfev),njev=int(solution.njev),nlu=int(solution.nlu),
                accepted_intervals=len(solution.t)-1,success=bool(solution.success),
                endpoint=float(solution.y[0,-1]),error=abs(float(solution.y[0,-1])-math.cos(1))))
    results["gradients"]=[scalar_gradient(*case,steps,method)
        for case in [(-.7,1.2,.4,1.3),(.3,.8,1.,.7)]
        for steps in [3,4,16,64] for method in ["euler","rk4"]]
    results["reverse_conditioning"]=dict(rate=-20.,endpoint=1.,
        exact_final=math.exp(-20.),absolute_final_perturbation=1e-8,
        recovered_initial=(math.exp(-20.)+1e-8)*math.exp(20.),
        amplification=math.exp(20.))
    results["topology"]=[dict(initial=values,endpoint=endpoint,threshold=threshold,
        augmented_final=[[value,endpoint*value*value] for value in values],
        predicted=[int(endpoint*value*value>threshold) for value in values])
        for values,endpoint,threshold in [([-1.,0.,1.],1.,.5),([-2.,0.,1.],.4,.5),([-2.,0.,1.],.7,.5)]]
    results["order_flip"]=dict(rate=-2.,step_size=1.,initial=[-1.,1.],
        euler_final=[1.,-1.],exact_final=[-math.exp(-2.),math.exp(-2.)])
    results["observations"]=[]
    for times,values,query in [([.2,.9,1.3],[1.,-.5,.8],1.6),([.1,.7,1.4],[1.,-1.,.5],1.6)]:
        results["observations"].append(dict(times=times,values=values,
            baseline=observations(times,values,query),
            omitted_middle=observations(times,values,query,1),
            zero_middle=observations(times,[values[0],0.,values[2]],query),
            query_before_last=observations(times,values,1.)))
    matrix=np.array([[.2,2.],[.4,-.1]])
    probes=[np.array([x,y]) for x in [-1.,1.] for y in [-1.,1.]]
    trace_estimates=[float(probe@matrix@probe) for probe in probes]
    assert abs(np.mean(trace_estimates)-np.trace(matrix))<1e-14
    results["density"]=dict(matrix=matrix.tolist(),trace=float(np.trace(matrix)),
        probe_values=trace_estimates,mean=float(np.mean(trace_estimates)),
        endpoint=2.,log_density_change=-2.*float(np.trace(matrix)),
        volume_ratio=float(np.linalg.det(expm(2.*matrix))),
        density_ratio=math.exp(-2.*float(np.trace(matrix))))
    # Two different conditional paths meet; least squares learns their mean velocity.
    results["flow_matching"]=dict(pairs=[[0.,2.],[2.,0.]],time=.5,
        positions=[1.,1.],target_velocities=[2.,-2.],conditional_mean=0.,
        mean_squared_loss_at_mean=4.,mean_squared_loss_at_first_velocity=8.)
    features,labels,roles,metadata=load_data()
    snapshots=json.loads((DIRECTORY/"fitted-models.json").read_text())
    results["real_input"]=[]
    parity=0.
    for kind in ["neural_ode","augmented_ode"]:
        snapshot=next(row for row in snapshots if row["kind"]==kind and row["seed"]==37)
        model=native_model(snapshot)
        for source_id in [64,70,109]:
            assert source_id-1 in roles["validation"]
            for delta in [0.,.6,-.6]:
                initial=features[source_id-1].detach().numpy().copy()
                initial[2]+=delta/metadata["scale"][2]
                variants=[]
                for method,steps in [("rk4",4),("rk4",16),("rk4",64),("euler",4)]:
                    independent=independent_inference(snapshot,initial,steps,method)
                    with torch.no_grad():
                        logits,trace=model(torch.tensor(initial[None,:]),steps,method,return_trace=True)
                    discrepancy=max(float(np.max(np.abs(np.array(independent["logits"])-logits[0].numpy()))),
                        float(np.max(np.abs(np.array(independent["trace"])-trace[:,0].numpy()))))
                    parity=max(parity,discrepancy)
                    variants.append(dict(method=method,steps=steps,**independent))
                tensor=torch.tensor(initial[None,:],requires_grad=True)
                selected_probability=torch.softmax(model(tensor),-1)[0,1]
                derivative=torch.autograd.grad(selected_probability,tensor)[0][0,2].item()/metadata["scale"][2]
                epsilon=1e-5
                plus,minus=initial.copy(),initial.copy()
                plus[2]+=epsilon/metadata["scale"][2];minus[2]-=epsilon/metadata["scale"][2]
                finite=(independent_inference(snapshot,plus)["probabilities"][1]-
                        independent_inference(snapshot,minus)["probabilities"][1])/(2*epsilon)
                assert abs(derivative-finite)<2e-8
                results["real_input"].append(dict(kind=kind,seed=37,source_id=source_id,
                    true_class=int(labels[source_id-1]),petal_length_delta_cm=delta,
                    standardized_initial=initial.tolist(),variants=variants,
                    versicolor_probability_derivative_per_cm=derivative,central_difference=finite))
    assert parity<2e-12
    results["independent_network_max_error"]=parity
    results["majority_assessment"]=dict(class_count=[10,10,9],selected_class=0,
        correct=10,count=29,cross_entropy=math.log(3.))
    (DIRECTORY/"calculated-inputs.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(parity=parity,stiff=results["stiff_slow_manifold"],
        adaptive=[{key:row[key] for key in ["rates","tolerance","accepted","rejected","nfe"]} for row in results["adaptive"]],
        gradient_examples=[row for row in results["gradients"] if row["steps"] in [3,4]],
        real=[dict(kind=row["kind"],id=row["source_id"],delta=row["petal_length_delta_cm"],
              probability=row["variants"][0]["probabilities"]) for row in results["real_input"]]),indent=2))


if __name__=="__main__":
    main()
