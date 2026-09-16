"""Declared small WGAN comparison on real two-coordinate digit measurements."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
STEPS, CRITIC_STEPS, BATCH = 600, 3, 64
RATE, PENALTY, CLIP = .001, 10., .1
SEEDS = (11,29,47)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2,24),nn.Linear(24,24),nn.Linear(24,2)])

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = torch.relu(layer(z))
        return torch.sigmoid(self.layers[-1](z))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2,24),nn.Linear(24,24),nn.Linear(24,1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.leaky_relu(layer(x),negative_slope=.2)
        return self.layers[-1](x).squeeze(-1)


def gradient_penalty(critic, real, fake, rng):
    shape = (len(real),)+(1,)*(real.ndim-1)
    mixing = torch.rand(shape,generator=rng,dtype=real.dtype,device=real.device)
    points = (mixing*real.detach()+(1-mixing)*fake.detach()).requires_grad_(True)
    values = critic(points)
    gradients = torch.autograd.grad(values.sum(),points,create_graph=True)[0]
    norms = gradients.flatten(1).norm(dim=1)
    return ((norms-1)**2).mean()


def projected_distance(left, right):
    # A fixed finite directional average, not exact multivariate Wasserstein or FID.
    angles = np.arange(64)*np.pi/64
    directions = np.c_[np.cos(angles),np.sin(angles)]
    a,b = np.asarray(left)@directions.T, np.asarray(right)@directions.T
    return float(np.mean([wasserstein_distance(a[:,i],b[:,i]) for i in range(64)]))


def effective_layers(model):
    return [dict(weight=l.weight.detach().double().tolist(),bias=l.bias.detach().double().tolist())
            for l in model.layers]


def main():
    records = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    pixels = np.array([[int(row[f'pixel_{j}']) for j in range(64)] for row in records])
    ids = np.array([int(row['source_id']) for row in records])
    image = pixels.reshape(-1,8,8)
    left = image[:,:,:4].sum((1,2)); right = image[:,:,4:].sum((1,2))
    values = np.c_[left,right]/512.  # each half:32 pixels, known max16
    # Equal two-coordinate measurements stay in one role; their frequency is retained.
    grouped = {}
    for row,pair in enumerate(zip(left,right)):
        grouped.setdefault(tuple(int(v) for v in pair),[]).append(row)
    keys = list(grouped)
    order = np.random.default_rng(91).permutation(len(keys))
    first,second = int(.6*len(keys)),int(.8*len(keys))
    group_roles = dict(fit=order[:first],development=order[first:second],assessment=order[second:])
    roles = {name: np.array([i for k in groups for i in grouped[keys[k]]])
             for name,groups in group_roles.items()}
    data = torch.tensor(values,dtype=torch.float32)
    fit = data[roles['fit']]
    evaluation_z = torch.randn(256,2,generator=torch.Generator().manual_seed(2026))
    bootstrap = values[np.random.default_rng(2026).choice(roles['fit'],256,replace=True)]
    baseline = {name: projected_distance(bootstrap,values[rows]) for name,rows in roles.items()}
    results = []
    for method in ('clipping','gradient-penalty','spectral-normalization'):
        for seed in SEEDS:
            torch.manual_seed(seed)
            generator,critic = Generator(),Critic()
            if method=='spectral-normalization':
                for layer in critic.layers:
                    spectral_norm(layer,n_power_iterations=1)
            if method=='clipping':
                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-CLIP,CLIP)
            opt_g = torch.optim.Adam(generator.parameters(),lr=RATE,betas=(0.,.9))
            opt_d = torch.optim.Adam(critic.parameters(),lr=RATE,betas=(0.,.9))
            draws = torch.Generator().manual_seed(seed+1000)
            gp_draws = torch.Generator().manual_seed(seed+2000)
            history = []
            for step in range(1,STEPS+1):
                critic.train()
                critic.requires_grad_(True)
                for _ in range(CRITIC_STEPS):
                    real = fit[torch.randint(len(fit),(BATCH,),generator=draws)]
                    with torch.no_grad():
                        fake = generator(torch.randn(BATCH,2,generator=draws))
                    # One joined forward: both groups use the same current SN weight.
                    scores = critic(torch.cat((real,fake)))
                    score_real,score_fake = scores[:BATCH],scores[BATCH:]
                    penalty = gradient_penalty(critic,real,fake,gp_draws) if method=='gradient-penalty' else scores.new_zeros(())
                    critic_loss = score_fake.mean()-score_real.mean()+PENALTY*penalty
                    opt_d.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    opt_d.step()
                    if method=='clipping':
                        with torch.no_grad():
                            for parameter in critic.parameters():
                                parameter.clamp_(-CLIP,CLIP)
                # Freeze critic parameters and spectral-vector updates, not input derivatives.
                critic.eval()
                critic.requires_grad_(False)
                generated = generator(torch.randn(BATCH,2,generator=draws))
                generator_loss = -critic(generated).mean()
                opt_g.zero_grad(set_to_none=True)
                generator_loss.backward()
                opt_g.step()
                if step in (1,100,300,600):
                    with torch.no_grad():
                        generated_eval = generator(evaluation_z).numpy()
                    history.append(dict(step=step,critic_loss=float(critic_loss.detach()),
                                        generator_loss=float(generator_loss.detach()),
                                        raw_gp=float(penalty.detach()),
                                        development_projected_w1=projected_distance(generated_eval,values[roles['development']]),
                                        generated=generated_eval.tolist()))
            critic.eval()
            with torch.no_grad():
                generated = generator(evaluation_z)
                critic_values = critic(data)
                singular_values = [torch.linalg.svdvals(l.weight).tolist() for l in critic.layers]
            probe = data[roles['assessment']].clone().requires_grad_(True)
            grad = torch.autograd.grad(critic(probe).sum(),probe)[0]
            grad_norm = grad.norm(dim=1)
            grid_values = torch.linspace(0,1,41)
            grid = torch.cartesian_prod(grid_values,grid_values).requires_grad_(True)
            grid_score = critic(grid)
            grid_grad = torch.autograd.grad(grid_score.sum(),grid)[0]
            # Save a fresh latent-input edit for the later frozen-model investigation.
            latent_pair = torch.tensor([[-.7,.4],[-.7,1.1]])
            with torch.no_grad():
                edited_outputs = generator(latent_pair)
            result = dict(method=method,seed=seed,history=history,
                          generator_parameters=sum(p.numel() for p in generator.parameters()),
                          critic_parameters=sum(p.numel() for p in critic.parameters()),
                          metrics={name: projected_distance(generated.numpy(),values[rows]) for name,rows in roles.items()},
                          generated=generated.tolist(),generator_layers=effective_layers(generator),
                          critic_layers=effective_layers(critic),critic_values=critic_values.tolist(),
                          critic_singular_values=singular_values,
                          matrix_product_bound=float(np.prod([v[0] for v in singular_values])),
                          assessment_gradient_norm=grad_norm.tolist(),
                          assessment_max_gradient=float(grad_norm.max()),
                          grid_coordinates=grid.detach().tolist(),grid_score=grid_score.detach().tolist(),
                          grid_gradient=grid_grad.detach().tolist(),grid_max_gradient=float(grid_grad.norm(dim=1).max()),
                          latent_intervention=dict(inputs=latent_pair.tolist(),outputs=edited_outputs.tolist()))
            results.append(result)
            print(method,seed,'projected W1',round(result['metrics']['assessment'],6),
                  'sample gradient max',round(result['assessment_max_gradient'],6),
                  'matrix product',round(result['matrix_product_bound'],6),flush=True)
    output = dict(protocol=dict(torch=torch.__version__,numpy=np.__version__,steps=STEPS,critic_steps=CRITIC_STEPS,
                               batch=BATCH,rate=RATE,penalty=PENALTY,clip=CLIP,seeds=SEEDS,rows=len(data),
                               unique_profile_groups=len(keys),roles={k:ids[v].tolist() for k,v in roles.items()},
                               collision_groups=[ids[g].tolist() for g in grouped.values() if len(g)>1],
                               csv_sha256=hashlib.sha256((ROOT/'digits-400.csv').read_bytes()).hexdigest()),
                  measurements=values.tolist(),source_ids=ids.tolist(),evaluation_latents=evaluation_z.tolist(),
                  bootstrap=dict(generated=bootstrap.tolist(),metrics=baseline),fits=results)
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('roles',{k:len(v) for k,v in roles.items()},'unique groups',len(keys),'baseline',baseline,flush=True)


if __name__=='__main__':
    main()
