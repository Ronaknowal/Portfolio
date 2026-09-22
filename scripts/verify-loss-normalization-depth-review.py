"""Complementary source-bound review, independent of the two program authors."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/teaching/implementation-depth/loss-normalization-independent.json'
LOSS = 'public/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-mechanisms.py'
NORM = 'public/learn-assets/batch-layer-group-rms-normalization/normalization-backward.py'
sources = [LOSS, NORM, 'scripts/verify-loss-normalization-depth-review.py',
           'src/learn/data/loss-mechanisms-program.js',
           'src/learn/data/normalization-backward-program.js']
for topic in ('loss-functions-ce-mse-focal-contrastive-triplet', 'batch-layer-group-rms-normalization'):
    sources.extend([f'docs/teaching/drafts/{topic}/lesson.md', f'src/learn/data/topics/{topic}.jsx'])
receipt = {'status':'running', 'reviewer':'Backpropagation agent, independent of new Loss/Normalization author',
           'sourceHashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}, 'checks':[]}
OUT.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def close(actual, expected, name, atol=2e-7, rtol=2e-7):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    receipt['checks'].append(name)

def directional(function, arrays, gradient, name, rng, step=1e-6):
    directions = [rng.normal(size=value.shape) for value in arrays]
    plus = function(*[value + step * d for value, d in zip(arrays, directions)])[0]
    minus = function(*[value - step * d for value, d in zip(arrays, directions)])[0]
    expected = sum(float(np.sum(g*d)) for g, d in zip(gradient, directions))
    close((plus-minus)/(2*step), expected, name)

def review():
    torch.set_num_threads(1)
    losses, norm = load('review_losses', LOSS), load('review_norm', NORM)
    rng = np.random.default_rng(31)
    for canonical, wrapper in ((LOSS, 'src/learn/data/loss-mechanisms-program.js'),
                               (NORM, 'src/learn/data/normalization-backward-program.js')):
        match = re.search(r'export default (.*);\s*$', (ROOT / wrapper).read_text(encoding='utf-8'), re.S)
        assert match and json.loads(match.group(1)) == (ROOT / canonical).read_text(encoding='utf-8')
        receipt['checks'].append(wrapper + ': displayed code equals complete canonical program')
    # Execute both complete programs and compare their served execution records.
    for module, filename in ((losses, LOSS), (norm, NORM)):
        output = io.StringIO()
        with contextlib.redirect_stdout(output): module.main()
        observed = json.loads(output.getvalue())
        expected = json.loads((ROOT / filename.replace('.py', '-output.json')).read_text())
        assert observed == expected, filename + ' saved output mismatch'
        receipt['checks'].append(filename + ': complete program reproduces its served output')

    prediction = np.array([[-1.3, .4, 2.1], [.2, 1.7, -2.2]])
    target = np.array([[.1, .2, .5], [-.3, .6, -1.]])
    for kind in ('mse', 'mae', 'huber', 'quantile'):
        function = lambda p: losses.regression(p, target, kind, delta=.37, quantile=.21)
        _, gradient = function(prediction)
        directional(function, [prediction], [gradient], 'changed matrix ' + kind + ' derivative/reduction', rng)

    logits = rng.normal(size=(3, 4)); labels = np.array([3, 0, 2])
    value, gradient = losses.cross_entropy(logits, labels)
    directional(lambda z: losses.cross_entropy(z, labels), [logits], [gradient], 'CE directional derivative on changed 3 by 4 logits', rng)
    shifted, shifted_gradient = losses.cross_entropy(logits + np.array([[100.], [-3.], [10.]]), labels)
    close([shifted], [value], 'CE per-row translation invariance', atol=1e-13, rtol=1e-13)
    close(shifted_gradient, gradient, 'CE shifted gradient identity', atol=1e-13, rtol=1e-13)
    close(gradient.sum(1), np.zeros(3), 'CE gradient row sums', atol=1e-14, rtol=0)
    for gamma in (0., .5, 2.):
        z=np.array([-30., -2., .1, 10.]); y=np.array([1., 0., 1., 0.])
        function=lambda p: losses.binary_focal(p, y, gamma)
        _, gradient=function(z)
        directional(function,[z],[gradient],f'focal gamma {gamma} changed directional derivative',rng)
    extreme, extreme_grad = losses.binary_focal(np.array([-1000., 1000., -1000., 1000.]), np.array([1.,0.,0.,1.]), 2.)
    close(extreme, 500., 'focal stable tails retain two hard errors', atol=0, rtol=0)
    close(extreme_grad, [-.25,.25,0.,0.], 'focal stable extreme gradients', atol=0, rtol=0)
    a=rng.normal(size=(3, 4)); p=rng.normal(size=(3, 4)); n=rng.normal(size=(3, 4))
    function=lambda x,y,z:losses.squared_triplet(x,y,z,margin=.4)
    _, gradients=function(a,p,n)
    directional(function,[a,p,n],gradients,'triplet all three inputs and changed margin',rng)
    q=rng.normal(size=(3, 4)); k=rng.normal(size=(3, 4))
    function=lambda x,y:losses.paired_info_nce(x,y,temperature=.7)
    value, gradients=function(q,k)
    directional(function,[q,k],gradients,'InfoNCE both raw feature derivatives',rng)
    scales=np.array([[2.],[.5],[3.]])
    scaled_value, scaled_gradients=function(q*scales,k)
    close(scaled_value,value,'InfoNCE positive query rescaling invariant',atol=1e-12,rtol=1e-12)
    close(scaled_gradients[0]*scales,gradients[0],'InfoNCE gradient inverse rescaling',atol=1e-12,rtol=1e-12)
    close((gradients[0]*q).sum(1),np.zeros(3),'InfoNCE raw gradient tangent to normalization sphere',atol=1e-12,rtol=0)
    value, gradients=losses.paired_info_nce(q[:1],k[:1])
    close(value,0.,'one candidate InfoNCE loss is zero',atol=0,rtol=0)
    close(np.concatenate([g.ravel() for g in gradients]),np.zeros(8),'one candidate InfoNCE gradients are zero',atol=0,rtol=0)
    tiny=np.array([[1e-14,0.],[0.,1e-14]])
    own, gradients=losses.paired_info_nce(tiny,tiny)
    tq=torch.tensor(tiny,requires_grad=True);tk=torch.tensor(tiny,requires_grad=True)
    exact=F.cross_entropy(F.normalize(tq,dim=1,eps=0.) @ F.normalize(tk,dim=1,eps=0.).T/.2,torch.arange(2))
    exact.backward()
    close(own,exact.item(),'tiny nonzero InfoNCE matches explicit no-floor reference',atol=1e-14,rtol=1e-14)
    close(gradients[0],tq.grad.numpy(),'tiny nonzero query gradient uses the matching contract',atol=.01,rtol=1e-12)
    # Implement the stated smoothing extension independently, then compare the API.
    z=rng.normal(size=(4,3)); targets=np.array([0,2,1,0])
    shifted=z-z.max(1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
    for epsilon in (0.,.2,1.):
        value=-(1-epsilon)*logp[np.arange(4),targets].mean()-epsilon*logp.mean()
        gradient=np.exp(logp)-epsilon/3;gradient[np.arange(4),targets]-=1-epsilon;gradient/=4
        tensor=torch.tensor(z,requires_grad=True)
        reference=F.cross_entropy(tensor,torch.tensor(targets),label_smoothing=epsilon);reference.backward()
        close(value,reference.item(),f'smoothing {epsilon} loss')
        close(gradient,tensor.grad.numpy(),f'smoothing {epsilon} gradient')

    # Noncontiguous grouped axes and a large epsilon detect missing scale terms.
    x=rng.normal(size=(2,3,4)); upstream=rng.normal(size=x.shape)
    for centered in (True,False):
        for axes in ((0,2),(2,)):
            for epsilon in (1e-4,.1,2.):
                _,cache=norm.normalize(x,axes,epsilon=epsilon,centered=centered)
                gradient=norm.backward(upstream,cache)
                function=lambda v:(float((norm.normalize(v,axes,epsilon=epsilon,centered=centered)[0]*upstream).sum()),None)
                directional(function,[x],[gradient],f'normalization directional centered={centered} axes={axes} eps={epsilon}',rng)
                if centered: close(gradient.sum(axis=axes),np.zeros_like(gradient.sum(axis=axes)),f'centered input gradient sums axes={axes} eps={epsilon}',atol=1e-12,rtol=0)
    constant=np.full((2,3),4.); incoming=np.array([[1.,-2.,.5],[0.,2.,1.]])
    h,cache=norm.normalize(constant,(1,),epsilon=.04)
    close(h,np.zeros((2,3)),'constant centered input outputs zero',atol=0,rtol=0)
    close(norm.backward(incoming,cache),(incoming-incoming.mean(1,keepdims=True))/ .2,'constant group retains centered incoming derivative',atol=1e-14,rtol=0)
    _,cache=norm.normalize(np.zeros((2,3)),(1,),epsilon=.04,centered=False)
    close(norm.backward(incoming,cache),incoming/.2,'zero RMS input derivative is inverse epsilon scale',atol=1e-14,rtol=0)
    _,cache=norm.normalize(np.array([[3.],[-2.]]),(1,),epsilon=.04)
    close(norm.backward(np.array([[1.],[2.]]),cache),np.zeros((2,1)),'singleton centered collection derivative',atol=0,rtol=0)
    h,cache=norm.normalize(x,(2,),epsilon=.1)
    gamma=np.array([.5,-.7,1.3,.2]);beta=np.array([.1,.2,-.3,0.]); lr=.03
    dg=(upstream*h).sum((0,1));db=upstream.sum((0,1))
    updated=(gamma-lr*dg)*h+(beta-lr*db)
    layer=torch.nn.LayerNorm(4,eps=.1,dtype=torch.float64)
    with torch.no_grad(): layer.weight.copy_(torch.tensor(gamma));layer.bias.copy_(torch.tensor(beta))
    optimizer=torch.optim.SGD(layer.parameters(),lr=lr)
    (layer(torch.tensor(x))*torch.tensor(upstream)).sum().backward();optimizer.step()
    close(updated,layer(torch.tensor(x)).detach().numpy(),'manual affine update extension matches actual SGD module',atol=1e-12,rtol=1e-12)
    receipt['runtime']={'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__}

try:
    review()
    receipt['status']='passed'
    receipt['count']=len(receipt['checks'])
except Exception as error:
    receipt['status']='failed';receipt['failure']=str(error)
    raise
finally:
    OUT.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(f"Independent Loss/Normalization depth review: {receipt['count']} checks passed.")
