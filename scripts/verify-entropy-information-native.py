import contextlib, io, itertools, json, math, platform, subprocess, sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
import mpmath as mp
import scipy
from scipy import integrate, optimize, special, stats
import torch
import torch.nn.functional as F

out=Path('scratch/entropy-verification')
data=json.loads((out/'cases.json').read_text(encoding='utf-8'))
cases=data['cases']
def value(x): return float(x) if isinstance(x,str) else x
def close(actual,expected,tol=2e-11):
    actual=value(actual)
    if math.isinf(expected): assert actual==expected,(actual,expected)
    else: assert math.isclose(actual,float(expected),abs_tol=tol,rel_tol=tol),(actual,expected)
for item in cases['information']:
    p=np.array(item['p'],dtype=float);p/=p.sum()
    q=np.array(item['q'],dtype=float);q/=q.sum()
    scale=math.log(item['base'])
    state=item['state']
    entropy=stats.entropy(p,base=item['base'])
    kl=special.rel_entr(p,q).sum()/scale
    ce=-special.xlogy(p,q).sum()/scale
    close(state['entropy'],entropy);close(state['kl'],kl);close(state['crossEntropy'],ce)
    for i,row in enumerate(state['rows']):
        close(row['entropy'],special.entr(p[i])/scale)
        close(row['crossEntropy'],-special.xlogy(p[i],q[i])/scale)
        close(row['kl'],special.rel_entr(p[i],q[i])/scale)
    assert kl>=-1e-12
mp.mp.dps=70
for item in data['nearMatch']:
    p=[mp.mpf(x) for x in item['state']['p']];q=[mp.mpf(x) for x in item['state']['q']]
    expected=sum(x*mp.log(x/y) for x,y in zip(p,q))
    actual=item['state']['kl']
    assert actual >= 0
    assert abs(mp.mpf(actual)-expected) <= abs(expected)*mp.mpf('1e-11')+mp.mpf('1e-50'), (actual,expected)
for state in cases['binary']:
    p=state['probability']
    close(state['entropy'],stats.entropy([p,1-p],base=2))
    for x,y in state['curve']:close(y,stats.entropy([x,1-x],base=2))
for item in cases['prefix']:
    state=item['state'];words=data['codebooks'][item['codebook']]['words']
    code=dict(zip('ABCD',words))
    lengths=[len(code[x]) for x in state['message']]
    end_positions=list(itertools.accumulate(lengths))
    completed=sum(end<=state['consumed'] for end in end_positions)
    assert state['decoded']==state['message'][:completed]
    expected_bits=''.join(code[x] for x in state['message'])
    assert state['bits']==expected_bits
    previous=end_positions[completed-1] if completed else 0
    assert state['buffer']==expected_bits[previous:state['consumed']]
    close(state['expectedLength'],sum(p*len(word) for p,word in zip([.5,.25,.125,.125],words)))
    assert sum(Fraction(1,2**len(word)) for word in words)==1
for state in cases['conditional']:
    e=Fraction(str(state['noise']));trust=state['trust']
    joint=np.array([[float((1-e)/2),float(e/2)],[float(e/2),float((1-e)/2)]])
    h_joint=stats.entropy(joint.ravel(),base=2)
    close(state['conditionalEntropy'],h_joint-1)
    close(state['jointEntropy'],h_joint)
    predicted=np.array([[trust,1-trust],[1-trust,trust]])
    ce=-special.xlogy(joint,predicted).sum()/math.log(2)
    close(state['modelLoss'],ce)
    close(state['excess'],ce-(h_joint-1))
    for row in state['rows']:close(row['mass'],joint[row['x'],row['y']])
logits=torch.tensor([[row['logit'] for row in state['rows']] for state in cases['logits']],dtype=torch.float64)
labels=torch.tensor([state['target'] for state in cases['logits']],dtype=torch.long)
losses=F.cross_entropy(logits,labels,reduction='none').numpy()
logp=F.log_softmax(logits,dim=1).numpy()
for i,state in enumerate(cases['logits']):
    close(state['loss'],losses[i])
    for j,row in enumerate(state['rows']):close(row['logProbability'],logp[i,j]);close(row['probability'],math.exp(logp[i,j]))
quadratures=0
for state in cases['continuous']:
    for coordinate in ['original','transformed']:
        row=state[coordinate];w=row['width']
        entropy=integrate.quad(lambda x:-math.log2(1/w)/w,0,w)[0]
        probability=integrate.quad(lambda x:1/w,*row['event'])[0]
        quadratures+=2
        close(row['entropy'],entropy);close(row['probability'],probability)
        close(row['entropy']-math.log2(row['binWidth']),state['quantizedEntropy'])
    close(state['original']['entropy']+math.log2(state['scale']),state['transformed']['entropy'])
root_cases=0
for state in cases['maximum']:
    m=state['mean'];q=np.array(state['q']);p=np.array(state['optimum'])
    close(q.sum(),1);close(q@[0,1,2],m);assert q.min()>=0
    close(p.sum(),1);close(p@[0,1,2],m)
    close(state['entropy'],stats.entropy(q,base=2));close(state['maximumEntropy'],stats.entropy(p,base=2))
    close(state['gap'],special.rel_entr(q,p).sum()/math.log(2))
    close(state['gap'],state['maximumEntropy']-state['entropy'])
    if 0<m<2:
        # Independent numerical root of the mean, without solving the author's quadratic.
        eta=optimize.brentq(lambda z: np.dot(special.softmax(z*np.arange(3)),np.arange(3))-m,-50,50)
        reference=special.softmax(eta*np.arange(3))
        np.testing.assert_allclose(p,reference,atol=1e-12,rtol=1e-12)
        lower=max(0,m-1);upper=m/2
        # Optimize a unit interval so the optimizer's absolute position tolerance
        # does not swallow a very narrow feasible slice near a boundary mean.
        def negative_entropy_at_fraction(fraction):
            t=lower+fraction*(upper-lower)
            return -stats.entropy([max(0,1-m+t),max(0,m-2*t),t],base=2)
        optimum=optimize.minimize_scalar(negative_entropy_at_fraction,bounds=(0,1),method='bounded',options={'xatol':1e-13})
        close(state['maximumEntropy'],-optimum.fun,3e-10)
        root_cases+=1
    for t,h in state['curve']:close(h,stats.entropy([max(0,1-m+t),max(0,m-2*t),t],base=2))
# Execute the actual displayed complete programs, not retyped copies.
namespace={}
for key,example in data['examples'].items():
    result=subprocess.run([sys.executable,'-c',example['code']],text=True,capture_output=True,check=True)
    assert result.stdout.rstrip()==example['expected'],(key,result.stdout,example['expected'])
    with contextlib.redirect_stdout(io.StringIO()):
        scope={};exec(compile(example['code'],key,'exec'),scope)
    namespace[key]=scope
# Change the actual native helper inputs; compare independent library results.
native_cases=0
for item in cases['information'][::17]:
    p=np.array(item['p'],dtype=float);p/=p.sum();q=np.array(item['q'],dtype=float);q/=q.sum()
    h,c,k=namespace['support']['information'](p.tolist(),q.tolist(),item['base'])
    close(h,stats.entropy(p,base=item['base']));close(c,-special.xlogy(p,q).sum()/math.log(item['base']));close(k,special.rel_entr(p,q).sum()/math.log(item['base']));native_cases+=1
for code in [{'A':'0','B':'10','C':'110','D':'111'},{'A':'00','B':'01','C':'10','D':'11'},{'A':'00','B':'010','C':'1'}]:
    for n in range(5):
        for symbols in itertools.product(code,repeat=n):
            message=''.join(symbols);bits=''.join(code[x] for x in symbols)
            assert namespace['prefix']['decode'](bits,code)==message;native_cases+=1
for bits,code in [('11',{'A':'0','B':'10','C':'110','D':'111'}),('00',{'A':'0','B':'00','C':'1'}),('2',{'A':'0','B':'1'}),('1',{'A':'0'}),('',{'A':'','B':'1'})]:
    try:namespace['prefix']['decode'](bits,code);raise AssertionError('Invalid input accepted')
    except ValueError:pass
for length in [2,3,5,13]:
    for shift in [-1000,0,1000]:
        scores=np.linspace(-800,800,length)+shift
        actual=namespace['logits']['log_probabilities'](scores.tolist())
        np.testing.assert_allclose(actual,special.log_softmax(scores),atol=1e-12,rtol=1e-12);native_cases+=1
for mean_p,sd_p,mean_q,sd_q in itertools.product([-2,0,3],[.5,1,2],[-1,0,4],[.25,1,3]):
    expected=integrate.quad(lambda x:stats.norm.pdf(x,mean_p,sd_p)*(stats.norm.logpdf(x,mean_p,sd_p)-stats.norm.logpdf(x,mean_q,sd_q)),-np.inf,np.inf)[0]
    actual=namespace['continuous']['normal_kl'](mean_p,sd_p,mean_q,sd_q)
    close(actual,expected,1e-9);quadratures+=1;native_cases+=1
# Independent changed practice and proof counterexamples.
assert namespace['prefix']['decode']('110010111',namespace['prefix']['CODE'])=='CABD'
close(stats.entropy([.5,.125,.125,.125,.125],base=2),2)
close(stats.entropy([.75,.25],[.5,.5],base=2),.18872187554086717)
close(stats.entropy([.5,.5],[.75,.25],base=2),.20751874963942185)
assert stats.entropy([.1,.9],[.9,.1])>stats.entropy([.1,.9],[.5,.5])+stats.entropy([.5,.5],[.9,.1])
close(-np.log([.9]*4+[.4]*2).mean(),.37567058772993583)
close(-np.log([.6]*5+[.01]).mean(),1.1932163841363408)
close(stats.entropy([.35,.65],base=2)-stats.entropy([.2,.8],base=2),.21213996048812865)
result={'passed':True,'timestamp':datetime.now(timezone.utc).isoformat(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},'cases':{key:len(items) for key,items in cases.items()},'nearMatchHighPrecisionCases':len(data['nearMatch']),'invalidCases':data['invalidCases'],'completePrograms':len(data['examples']),'preservedOriginalPrograms':data['originalPreserved'],'additionalNativeHelperCases':native_cases,'independentConstrainedRootsAndOptimizers':root_cases,'independentQuadratures':quadratures}
(out/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
