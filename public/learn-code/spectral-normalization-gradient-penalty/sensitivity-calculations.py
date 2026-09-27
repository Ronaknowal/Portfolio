"""Exact small sensitivity examples and frozen-model checks; no GAN refitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations

ROOT = Path(__file__).resolve().parent


def power_trace(matrix, initial, steps=8):
    matrix = np.asarray(matrix, dtype=float)
    u = np.asarray(initial, dtype=float)
    u /= np.linalg.norm(u)
    records = []
    exact = np.linalg.svd(matrix, compute_uv=False)[0]
    for step in range(1, steps+1):
        v = matrix.T @ u
        v /= np.linalg.norm(v)
        u = matrix @ v
        u /= np.linalg.norm(u)
        estimate = float(u @ matrix @ v)
        records.append(dict(step=step,u=u.tolist(),v=v.tolist(),estimate=estimate,
                            normalized_operator_norm=float(exact/estimate)))
    return records


def forward(points, layers, kind):
    value = np.asarray(points, dtype=float)
    for index, layer in enumerate(layers):
        value = value @ np.asarray(layer['weight']).T + np.asarray(layer['bias'])
        if index < len(layers)-1:
            value = np.maximum(value, 0) if kind=='generator' else np.where(value>=0,value,.2*value)
        elif kind=='generator':
            value = 1/(1+np.exp(-value))
    return value


def main():
    diagonal = np.diag([3.,1.])
    fresh = np.array([[2.,1.],[0.,1.]])
    convolution = np.array([[1.,1.,0.],[0.,1.,1.]])
    nonoverlap = np.array([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    circular = np.array([[1.,1.,0.,0.],[0.,1.,1.,0.],[0.,0.,1.,1.],[1.,0.,0.,1.]])
    conv_norms = {name: float(np.linalg.svd(matrix,compute_uv=False)[0]/np.sqrt(2))
                 for name,matrix in dict(valid=convolution,disjoint=nonoverlap,circular=circular).items()}
    weight = np.array([3.,4.]); strength = 2.; rate = .1
    penalty_gradient = 2*strength*(np.linalg.norm(weight)-1)*weight/np.linalg.norm(weight)
    updated = weight-rate*penalty_gradient
    jacobian = np.array([[.5,-.5],[-.5,.5]])
    # Differentiating through sigma differs from treating sigma as a constant.
    matrix = torch.tensor([[2.,1.],[0.,1.]],dtype=torch.float64,requires_grad=True)
    upstream = torch.tensor([[1.,-.3],[.2,.7]],dtype=torch.float64)
    sigma = torch.linalg.svdvals(matrix)[0]
    loss = ((matrix/sigma)*upstream).sum()
    analytical = torch.autograd.grad(loss,matrix)[0].numpy()
    source = matrix.detach().numpy()
    finite = np.zeros_like(source)
    def objective(value):
        return float((value/np.linalg.svd(value,compute_uv=False)[0]*upstream.numpy()).sum())
    for i in range(2):
        for j in range(2):
            offset = np.zeros_like(source); offset[i,j]=1e-5
            finite[i,j]=(objective(source+offset)-objective(source-offset))/(2e-5)
    torch.manual_seed(81)
    linear = spectral_norm(nn.Linear(2,3,dtype=torch.float64),n_power_iterations=1)
    linear.eval()
    inputs = torch.tensor([[1.,2.],[-.4,.7]],dtype=torch.float64)
    before = linear(inputs).detach().numpy()
    remove_parametrizations(linear,'weight',leave_parametrized=True)
    after = linear(inputs).detach().numpy()
    study = json.loads((ROOT/'calculated-inputs.json').read_text(encoding='utf-8'))
    checked = []
    for result in study['fits']:
        generated = forward(study['evaluation_latents'],result['generator_layers'],'generator')
        critic_values = forward(study['measurements'],result['critic_layers'],'critic').ravel()
        edited = forward(result['latent_intervention']['inputs'],result['generator_layers'],'generator')
        # A nontrivial symmetry null: change latent coordinates and first-layer columns together.
        swapped = json.loads(json.dumps(result['generator_layers']))
        swapped[0]['weight'] = np.asarray(swapped[0]['weight'])[:,::-1].tolist()
        joint_swap = forward(np.asarray(study['evaluation_latents'])[:,::-1],swapped,'generator')
        latent_only = forward(np.asarray(study['evaluation_latents'])[:,::-1],result['generator_layers'],'generator')
        checked.append(dict(method=result['method'],seed=result['seed'],
                            sample_error=float(np.max(np.abs(generated-result['generated']))),
                            critic_error=float(np.max(np.abs(critic_values-result['critic_values']))),
                            latent_edit_error=float(np.max(np.abs(edited-result['latent_intervention']['outputs']))),
                            joint_swap_error=float(np.max(np.abs(joint_swap-generated))),
                            latent_only_max_change=float(np.max(np.abs(latent_only-generated)))))
    output = dict(
        circle=dict(matrix=diagonal.tolist(),singular_values=[3.,1.],frobenius=float(np.sqrt(10)),
                    normalized=[[1.,0.],[0.,1/3]],fresh_matrix=fresh.tolist(),
                    fresh_singular_values=np.linalg.svd(fresh,compute_uv=False).tolist()),
        power=dict(generic=power_trace(diagonal,[1,1]),orthogonal=power_trace(diagonal,[0,1]),
                   slow_gap=power_trace(np.diag([1.01,1]),[1,1]),
                   rotated_weight_cached_vector=power_trace(np.diag([1.,3.]),[1,0])),
        convolution=dict(valid_matrix=convolution.tolist(),disjoint_matrix=nonoverlap.tolist(),
                         circular_matrix=circular.tolist(),kernel_norm=float(np.sqrt(2)),
                         normalized_operator_norms=conv_norms),
        penalty=dict(weight=weight.tolist(),strength=strength,initial=32.,gradient=penalty_gradient.tolist(),
                     learning_rate=rate,updated=updated.tolist(),updated_norm=float(np.linalg.norm(updated)),
                     updated_penalty=float(strength*(np.linalg.norm(updated)-1)**2),
                     norms=[0,.5,1,2],two_sided=[1,.25,0,1],one_sided=[0,0,0,1],zero_centered=[0,.25,1,4]),
        missed_region=dict(function='x+4*ReLU(x-1)',sampled_points=[-.5,0,.5],sampled_gradients=[1,1,1],
                           sampled_penalty=0,outside_point=2,outside_gradient=5,outside_unit_penalty=16,
                           fresh_knot=2,fresh_probe=1.5,fresh_probe_gradient=1),
        batch=dict(jacobian=jacobian.tolist(),gradient_of_sum=jacobian.sum(0).tolist(),self_derivatives=[.5,.5]),
        composition=dict(first=[[3.,0.],[0.,1/3]],second=[[1/3,0.],[0.,3.]],product_bound=9.,actual_norm=1.,
                         residual_branch_norm=.5,residual_total_norm=1.5),
        margin=dict(weights=[[1.,0.],[0.,1.]],input=[2.,0.],logits=[2.,0.],gap=2.,
                    joint_lipschitz=1.,pair_lipschitz=float(np.sqrt(2)),radius=float(np.sqrt(2)),
                    boundary_perturbation=[-1.,1.],boundary_logits=[1.,1.]),
        fresh_investigations=dict(
            convolution_normalized_norm=float(np.linalg.svd([[1.,2.,0.],[0.,1.,2.]],compute_uv=False)[0]/np.sqrt(5)),
            moved_probe_penalty=float(2*np.mean((np.array([1.,1.,5.])-1)**2)),
            margin_input=[1.5,.25],margin_gap=1.25,margin_radius=float(1.25/np.sqrt(2))),
        normalization_gradient=dict(autograd=analytical.tolist(),finite_difference=finite.tolist(),
                                    max_error=float(np.max(np.abs(analytical-finite)))),
        export=dict(max_error=float(np.max(np.abs(before-after)))),frozen_model_checks=checked)
    assert output['normalization_gradient']['max_error']<1e-8
    assert output['export']['max_error']==0
    assert max(r['sample_error'] for r in checked)<1e-5
    assert max(r['critic_error'] for r in checked)<1e-5
    assert max(r['joint_swap_error'] for r in checked)<1e-12
    (ROOT/'sensitivity-results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(convolution=conv_norms,penalty=output['penalty'],
                          gradient_error=output['normalization_gradient']['max_error'],frozen_checks=checked),indent=2))


if __name__=='__main__':
    main()
