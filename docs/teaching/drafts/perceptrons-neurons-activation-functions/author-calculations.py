"""Author calculations and small CPU comparison; implementation is deferred."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import sklearn

directory = Path(__file__).resolve().parent
torch.set_num_threads(1)
output = {'versions':{'python':'3.12.14','numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__}}
corners = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
output['perceptron'] = {}
for name,labels in [('and',[-1,-1,-1,1]),('xor',[-1,1,1,-1])]:
    augmented = np.column_stack([corners,np.ones(4)])
    weights = np.zeros(3)
    epochs = []
    for epoch in range(12):
        updates = 0
        for row,label in zip(augmented,labels):
            if label*(row@weights) <= 0:
                weights += label*row
                updates += 1
        epochs.append({'epoch':epoch+1,'weights':weights.tolist(),'updates':updates,
                       'predictions':np.where(augmented@weights>0,1,-1).tolist()})
        if updates==0:break
    output['perceptron'][name]=epochs
sum_inputs = corners.sum(axis=1)
output['xor'] = {'hidden':np.column_stack([np.maximum(sum_inputs,0),np.maximum(sum_inputs-1,0)]).tolist(),
                 'output':(np.maximum(sum_inputs,0)-2*np.maximum(sum_inputs-1,0)).tolist()}
z = torch.tensor([-5.,-2.,-1.,0.,1.,2.,5.],dtype=torch.float64,requires_grad=True)
functions={'sigmoid':torch.sigmoid,'tanh':torch.tanh,'relu':F.relu,
           'leaky_relu':lambda value:F.leaky_relu(value,.1),'elu':F.elu,
           'gelu':lambda value:F.gelu(value,approximate='none'),
           'gelu_tanh':lambda value:F.gelu(value,approximate='tanh'),
           'silu':F.silu,'mish':F.mish}
output['activations']={'z':z.detach().tolist(),'functions':{}}
for name,function in functions.items():
    values=function(z)
    derivatives=torch.autograd.grad(values.sum(),z)[0]
    output['activations']['functions'][name]={'values':values.detach().tolist(),'derivatives':derivatives.tolist()}
data=np.genfromtxt(directory/'digits-400.csv',delimiter=',',names=True)
X=np.column_stack([data[f'pixel_{index}'] for index in range(64)]).astype(np.float32)/16
y=data['digit'].astype(np.int64)
train,valid=train_test_split(np.arange(400),test_size=120,stratify=y,random_state=22)
Xtrain=torch.tensor(X[train]);Ytrain=torch.tensor(y[train])
Xvalid=torch.tensor(X[valid]);Yvalid=torch.tensor(y[valid])
output['digitSplits']={'trainSourceIds':data['source_id'][train].astype(int).tolist(),
                       'validationSourceIds':data['source_id'][valid].astype(int).tolist()}
output['digitComparison']=[]
for seed in [1,2,3]:
    for name,activation in [('sigmoid',nn.Sigmoid),('tanh',nn.Tanh),('relu',nn.ReLU),
                            ('leaky_relu',lambda:nn.LeakyReLU(.1)),('gelu',nn.GELU),('silu',nn.SiLU)]:
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(64,32),activation(),nn.Linear(32,10))
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        trace=[]
        for step in range(201):
            if step in [0,1,10,50,100,200]:
                with torch.no_grad():
                    trace.append({'step':step,'trainLoss':F.cross_entropy(model(Xtrain),Ytrain).item(),
                                  'validationAccuracy':(model(Xvalid).argmax(1)==Yvalid).float().mean().item()})
            if step==200:break
            optimizer.zero_grad()
            loss=F.cross_entropy(model(Xtrain),Ytrain)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits=model(Xvalid)
            row={'seed':seed,'activation':name,'trainLoss':F.cross_entropy(model(Xtrain),Ytrain).item(),
                 'trainAccuracy':(model(Xtrain).argmax(1)==Ytrain).float().mean().item(),
                 'validationAccuracy':(logits.argmax(1)==Yvalid).float().mean().item(),
                 'validationPredictions':logits.argmax(1).tolist(),'trace':trace}
        output['digitComparison'].append(row)

output['memory']={'singleHiddenElements':8*4096*16384,'bf16Bytes':2*8*4096*16384,
                  'plainWeights':2*4096*16384,'gatedExactBudgetInner':16384*2/3,
                  'gatedRoundedInner':256*math.ceil((16384*2/3)/256),
                  'gatedRoundedWeights':3*4096*(256*math.ceil((16384*2/3)/256))}
output['investigationFixtures'] = {
    'geometry': [],
    'xorRepairs': [],
    'localSensitivity': [],
}
for name,point,weight,bias in [
    ('worked', [2,-1], [1.5,-2], -1),
    ('scaled', [2,-1], [3,-4], -2),
    ('tie', [1,1], [1,-1], 0),
    ('tieScaled', [1,1], [2,-2], 0),
    ('positive', [1,0.5], [1,-1], 0),
    ('negative', [1,1.5], [1,-1], 0),
    ('zeroWeights', [1,1], [0,0], 1),
]:
    score=float(np.dot(point,weight)+bias)
    norm=float(np.linalg.norm(weight))
    output['investigationFixtures']['geometry'].append({
        'name':name,'point':point,'weight':weight,'bias':bias,'score':score,
        'distance':score/norm if norm else None,'hard':int(score>0),
        'sigmoid':1/(1+math.exp(-score)),
    })
for bias,coefficient in [(-1,-1),(-1,-2),(-.5,0),(-.5,-4/3),(-3,-1),(-3,-2)]:
    hidden=np.maximum(sum_inputs+bias,0)
    result=np.maximum(sum_inputs,0)+coefficient*hidden
    output['investigationFixtures']['xorRepairs'].append({
        'bias':bias,'coefficient':coefficient,'hidden2':hidden.tolist(),
        'output':result.tolist(),
    })
for function,score,weight in [('relu',2,.5),('leaky_relu',2,.5),
                              ('relu',-2,.5),('leaky_relu',-2,.5),
                              ('sigmoid',0,4),('silu',-2,1),
                              ('relu',0,1)]:
    state=torch.tensor(float(score),dtype=torch.float64,requires_grad=True)
    value=functions[function](state)
    derivative=torch.autograd.grad(value,state)[0].item()
    output['investigationFixtures']['localSensitivity'].append({
        'function':function,'score':score,'weight':weight,'value':value.item(),
        'slope':derivative,'sensitivity':weight*derivative,
    })
(directory/'calculated-inputs.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output['perceptron'],indent=2))
print([(row['activation'],row['seed'],round(row['trainLoss'],6),round(row['validationAccuracy'],6)) for row in output['digitComparison']])
print(output['memory'])
