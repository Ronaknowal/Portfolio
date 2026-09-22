"""Small CPU architecture investigation; adjacent attributed digits-400.csv is required."""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


class PlainOrResidual(nn.Module):
    def __init__(self, channels=12, residual=False):
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual = residual

    def forward(self, x):
        branch = self.second(F.relu(self.first(x)))
        return F.relu(branch + x if self.residual else branch)


class ParallelBranches(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        quarter = channels // 4
        self.point = nn.Conv2d(channels, quarter, 1)
        self.small = nn.Sequential(nn.Conv2d(channels, quarter, 1), nn.ReLU(),
                                   nn.Conv2d(quarter, quarter, 3, padding=1))
        self.large = nn.Sequential(nn.Conv2d(channels, quarter, 1), nn.ReLU(),
                                   nn.Conv2d(quarter, quarter, 5, padding=2))
        self.pool_projection = nn.Conv2d(channels, quarter, 1)

    def forward(self, x):
        branches = [self.point(x), self.small(x), self.large(x),
                    self.pool_projection(F.max_pool2d(x, 3, 1, 1))]
        return F.relu(torch.cat(branches, dim=1))


class InvertedGated(nn.Module):
    def __init__(self, channels=12, expansion=3):
        super().__init__()
        expanded = channels * expansion
        self.expand = nn.Conv2d(channels, expanded, 1)
        self.spatial = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.squeeze = nn.Linear(expanded, expanded // 4)
        self.excite = nn.Linear(expanded // 4, expanded)
        self.project = nn.Conv2d(expanded, channels, 1)

    def forward(self, x):
        expanded = F.silu(self.expand(x))
        spatial = F.silu(self.spatial(expanded))
        summary = spatial.mean(dim=(-2, -1))
        gate = torch.sigmoid(self.excite(F.relu(self.squeeze(summary))))
        branch = self.project(spatial * gate[:, :, None, None])
        return x + branch


class SmallClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.stem = nn.Conv2d(1, 12, 3, padding=1)
        # Construct the common tail/head first: their initial tensors match across kinds.
        self.tail = nn.Conv2d(12, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)
        if kind in ('plain', 'residual'):
            self.body = PlainOrResidual(residual=kind == 'residual')
        elif kind == 'parallel':
            self.body = ParallelBranches()
        elif kind == 'inverted_gated':
            self.body = InvertedGated()
        else:
            raise ValueError('unknown body')

    def features(self, x):
        stem = F.max_pool2d(F.relu(self.stem(x)), 2)
        body = self.body(stem)
        return F.max_pool2d(F.relu(self.tail(body)), 2)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(-2, -1)))


def model_cost(model, x):
    """Conv/linear multiply-accumulates only, per supplied batch (here one image)."""
    rows = []
    handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            def record(module, args, out, layer_name=name):
                terms = module.weight[0].numel()
                rows.append({'layer': layer_name, 'output_shape': list(out.shape),
                             'parameters': sum(p.numel() for p in module.parameters()),
                             'macs': int(out.numel() * terms)})
            handles.append(layer.register_forward_hook(record))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'macs': sum(row['macs'] for row in rows), 'layers': rows}


def score(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return {'cross_entropy': F.cross_entropy(logits, y).item(),
                'correct': int((logits.argmax(1) == y).sum()), 'count': len(y)}


def exact_examples():
    def conv(ci, co, k, h, groups=1, bias=True):
        weights = k*k*ci*co//groups
        return {'parameters': weights + (co if bias else 0), 'macs': weights*h*h}
    vgg_trunk = 0
    ci = 3
    for co, repeats in [(64,2),(128,2),(256,3),(512,3),(512,3)]:
        for _ in range(repeats):
            vgg_trunk += 9*ci*co+co
            ci = co
    vgg_head = [(512*7*7+1)*4096, (4096+1)*4096, (4096+1)*1000]
    # Constructed two-channel CAM; class weights include a negative contribution.
    features = np.array([[[1.,2.],[0.,3.]], [[0.,1.],[2.,1.]]])
    weights = np.array([2.,-1.])
    def cam_result(f, w=weights, bias=.5):
        cam = np.einsum('c,chw->hw', w, f)
        return {'map':cam.tolist(), 'score':float(cam.mean()+bias)}
    edited = features.copy()
    edited[1,1,0] = 6
    permuted = features[:,::-1,::-1]
    zeros = np.zeros_like(features)
    # SE toy: input channels have means a,b; hidden ReLU(a-b), logits [h,-h].
    def gate(a,b):
        h = max(a-b,0)
        return [float(1/(1+np.exp(-h))),float(1/(1+np.exp(h)))]
    return {
        'vgg16': {'trunk':vgg_trunk, 'head_layers':vgg_head,
                  'total':vgg_trunk+sum(vgg_head), 'gap512_head':(512+1)*1000,
                  'fc6_fp32_weights_grad_adam_bytes':vgg_head[0]*4*4},
        'standard_64_128':conv(64,128,3,14,bias=False),
        'separable_64_128':{'parameters':9*64+64*128,'macs':(9*64+64*128)*14*14},
        'residual_bottleneck256_64_weights':256*64+9*64*64+64*256,
        'plain_two256_weights':2*9*256*256,
        'inception_480_to32_to480_weights':480*32+25*32*480,
        'inception_direct480_weights':25*480*480,
        'se256_hidden16_parameters':256*16+16+16*256+256,
        'compound_steps':[{'phi':p,'depth':1.2**p,'width':1.1**p,'resolution':1.15**p,
                           'dense_parameter_factor':(1.2*1.1**2)**p,
                           'dense_mac_factor':(1.2*1.1**2*1.15**2)**p} for p in range(5)],
        'cam':{'features':features.tolist(),'weights':weights.tolist(),'bias':.5,
               'base':cam_result(features),'edited_negative_channel':cam_result(edited),
               'joint_spatial_permutation':cam_result(permuted),'zero_features':cam_result(zeros),
               'zero_weights':cam_result(features,np.zeros(2))},
        'se_toy':{'means_2_1':gate(2,1),'means_2_3':gate(2,3),'means_0_0':gate(0,0)},
        'shape_fixtures':{'parallel_channel_widths':[3,3,3,3],'output_channels':12,
                          'dense_growth_start8_add3_layers4':[8,11,14,17,20]},
        'changed_practice': {'conv32_64_k3_weights':9*32*64,
                              'two3_same32_weights':2*9*32*32,
                              'one5_same32_weights':25*32*32,
                              'gap128_classes7_params':(128+1)*7,
                              'flat7x7x128_classes7_params':(49*128+1)*7}}


def main():
    records = np.genfromtxt(HERE/'digits-400.csv', delimiter=',', names=True)
    pixels = np.column_stack([records[f'pixel_{i}'] for i in range(64)])
    source_ids = records['source_id'].astype(int)
    y = torch.tensor(records['digit'].astype(int))
    x = torch.tensor(pixels/16, dtype=torch.float32).reshape(-1,1,8,8)
    train, development = train_test_split(np.arange(len(y)), test_size=.3,
                                          random_state=22, stratify=y.numpy())
    if len(np.unique(pixels, axis=0)) != len(pixels) or len(np.unique(source_ids)) != len(pixels):
        raise ValueError('duplicate pixels or specimen IDs: revise the split before fitting')
    output = {'environment':{'python':platform.python_version(),'torch':torch.__version__,
                             'numpy':np.__version__,'sklearn':sklearn.__version__,'threads':1},
              'data_audit':{'rows':len(y),'unique_pixel_vectors':len(np.unique(pixels,axis=0)),
                            'unique_source_ids':len(np.unique(source_ids))},
              'train_source_ids':source_ids[train].tolist(),
              'development_source_ids':source_ids[development].tolist(),
              'exact':exact_examples(),'fits':[]}
    for seed in (1,2,3):
        for kind in ('plain','residual','parallel','inverted_gated'):
            torch.manual_seed(seed)
            model = SmallClassifier(kind)
            cost = model_cost(model, x[:1])
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            trace = []
            for step in range(401):
                if step in (0,1,25,100,200,400):
                    trace.append({'step':step,'train':score(model,x[train],y[train]),
                                  'development':score(model,x[development],y[development])})
                if step == 400:
                    break
                model.train()
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[train]),y[train])
                loss.backward()
                optimizer.step()
            row = {'seed':seed,'kind':kind,'cost':cost,'trace':trace}
            if seed == 1:
                with torch.no_grad():
                    logits = model(x[development])
                    wrong = np.flatnonzero((logits.argmax(1)!=y[development]).numpy())
                    selected = [0, int(wrong[0]) if len(wrong) else 1]
                    observations = []
                    for dev_position in selected:
                        index = development[dev_position]
                        maps = model.features(x[index:index+1])[0]
                        scores = model.head(maps.mean((-2,-1)))
                        cams = torch.einsum('kc,chw->khw',model.head.weight,maps)
                        reconstructed = cams.mean((-2,-1))+model.head.bias
                        observations.append({'source_id':int(source_ids[index]),'actual':int(y[index]),
                            'input':x[index,0].tolist(),'feature_maps':maps.tolist(),
                            'logits':scores.tolist(),'probabilities':scores.softmax(0).tolist(),
                            'class_maps':cams.tolist(),
                            'cam_reconstruction_max_error':float((reconstructed-scores).abs().max())})
                    row['observations'] = observations
                    row['head_weight'] = model.head.weight.tolist()
                    row['head_bias'] = model.head.bias.tolist()
            output['fits'].append(row)
            final = trace[-1]['development']
            print(seed,kind,cost['parameters'],cost['macs'],final['correct'],round(final['cross_entropy'],8))
    (HERE/'calculated-inputs.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
