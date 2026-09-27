"""Bounded author fixtures for frozen small fits and manuscript derivations."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from trajectory_memory_models import prepare, TrajectoryMemoryClassifier

def main():
    torch.set_num_threads(2)
    folder=Path(__file__).parent
    x,labels,roles,_=prepare(folder)
    arrays=np.load(folder/'trajectory-memory-fits.npz',allow_pickle=False)
    report={'source_id':int(roles[1][0]+1),'models':[]}
    row=x[roles[1][0]:roles[1][0]+1].clone()
    edit=row.clone();edit[0,22,0]=-edit[0,22,0]
    for kind in ['rwkv4','positive_kernel']:
        model=TrajectoryMemoryClassifier(kind)
        prefix=f'{kind}_seed17::'
        state={name:torch.from_numpy(arrays[prefix+name]) for name in model.state_dict()}
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            base=model(row);modified=model(edit);reverse=model(row.flip(1))
            first,saved=model.features(row[:,:22])
            second,_=model.features(row[:,22:],saved)
            forget,_=model.features(row[:,22:])
            carried=model.classifier(torch.cat([first,second],1).mean(1))
            reset_middle=model.classifier(torch.cat([first,forget],1).mean(1))
            future=row.clone();future[0,30,0]+=1
            full_feature,_=model.features(row);future_feature,_=model.features(future)
            prefix_error=float((full_feature[:,:30]-future_feature[:,:30]).abs().max())
            def result(logits):
                return {'prediction':int(logits.argmax(1)[0]+1),
                        'original_class_probability':float(logits.softmax(1)[0,labels[roles[1][0]]]),
                        'logits':logits[0].tolist()}
            report['models'].append({'kind':kind,'seed':17,'base':result(base),
                'coordinate_edit':result(modified),'reverse':result(reverse),
                'carry':result(carried),'forget_middle':result(reset_middle),
                'carry_logit_error':float((carried-base).abs().max()),'prefix_error':prefix_error,
                'reset_error':float((model(row)-base).abs().max())})
    fresh=x[19:20].clone()
    report['fresh_investigation']={'source_id':20,'cut_after':15,'models':[]}
    for kind in ['rwkv4','positive_kernel']:
        model=TrajectoryMemoryClassifier(kind)
        prefix=f'{kind}_seed17::'
        model.load_state_dict({name:torch.from_numpy(arrays[prefix+name]) for name in model.state_dict()})
        with torch.no_grad():
            first,saved=model.features(fresh[:,:15])
            second,_=model.features(fresh[:,15:],saved)
            forgotten,_=model.features(fresh[:,15:])
            fresh_edit=fresh.clone();fresh_edit[0,15,0]=-fresh_edit[0,15,0]
            variants={'base':model(fresh),'carry':model.classifier(torch.cat([first,second],1).mean(1)),
                      'forget':model.classifier(torch.cat([first,forgotten],1).mean(1)),
                      'edit_point16_x':model(fresh_edit),'reverse':model(fresh.flip(1))}
            report['fresh_investigation']['models'].append({'kind':kind,
                'outputs':{n:{'prediction':int(v.argmax(1)[0]+1),'class1_probability':float(v.softmax(1)[0,0]),
                              'logits':v[0].tolist()} for n,v in variants.items()},
                'carry_error':float((variants['base']-variants['carry']).abs().max())})
    memory=torch.tensor([[2.,7.]],dtype=torch.float64,requires_grad=True)
    key=torch.tensor([1.,0.],dtype=torch.float64);value=torch.tensor([5.],dtype=torch.float64)
    loss=.5*((memory@key-value)**2).sum();loss.backward()
    report['delta_gradient']={'loss':float(loss.detach()),'gradient':memory.grad.tolist(),
                              'step_half':(memory-.5*memory.grad).detach().tolist()}
    report['coordinate_edit']={'position':23,'x_before':float(row[0,22,0]),'x_after':float(edit[0,22,0]),
                               'y_unchanged':float(row[0,22,1])}
    report['half_life']={str(retention):float(np.log(.5)/np.log(retention)) for retention in [.5,.9,.99]}
    (folder/'investigation-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
