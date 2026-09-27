"""Bounded saved-state checks and fresh investigation inputs. Never fits a model."""
from pathlib import Path
import json
import runpy
import numpy as np
import torch
from scipy.special import erf, softmax

ROOT = Path(__file__).resolve().parent
study = runpy.run_path(str(ROOT / 'vision-study.py'))
mechanisms = runpy.run_path(str(ROOT / 'vision-mechanisms.py'))


def numpy_vit(images, state):
    def array(key): return np.asarray(state[key], dtype=float)
    def linear(x, key): return x @ array(key+'.weight').T + array(key+'.bias')
    def norm(x, key):
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*array(key+'.weight')+array(key+'.bias')
    batch=len(images)
    patches=[]
    for row in range(0,8,2):
        for column in range(0,8,2):
            local=images[:,:,row:row+2,column:column+2].reshape(batch,4)
            patches.append(local@array('patch.weight').reshape(32,4).T+array('patch.bias'))
    patches=np.stack(patches,1)+array('patch_position')
    cls=np.repeat(array('cls')+array('cls_position'),batch,axis=0)
    tokens=np.concatenate((cls,patches),axis=1)
    all_weights=[]
    for block in range(2):
        prefix=f'blocks.{block}'
        qkv=linear(norm(tokens,prefix+'.norm1'),prefix+'.qkv').reshape(batch,17,3,4,8).transpose(2,0,3,1,4)
        query,key,value=qkv
        weights=softmax(query@key.swapaxes(-1,-2)/np.sqrt(8),axis=-1)
        all_weights.append(weights)
        tokens=tokens+linear((weights@value).transpose(0,2,1,3).reshape(batch,17,32),prefix+'.output')
        hidden=linear(norm(tokens,prefix+'.norm2'),prefix+'.ffn.0')
        hidden=.5*hidden*(1+erf(hidden/np.sqrt(2)))
        tokens=tokens+linear(hidden,prefix+'.ffn.2')
    features=norm(tokens,'norm')
    return linear(features[:,0],'head'), features, all_weights


def main():
    torch.set_num_threads(1)
    data,_=study['load_data']()
    models=study['reload_models']()
    states=json.loads((ROOT/'vision-models.json').read_text())
    expected=json.loads((ROOT/'study-results.json').read_text())
    model=models['vit']
    images,labels=data['test']
    with torch.no_grad():
        for name,m in models.items():
            result=study['evaluate'](m,images,labels)
            assert result['predictions']==expected[name]['metrics']['test']['predictions']
        np_logits,np_features,np_attention=numpy_vit(images[:3].numpy(),states['vit'])
        features,attention=model.encode(images[:3],capture=True)
        error=float(np.max(np.abs(np_logits-model(images[:3]).numpy())))
        assert error < 1e-4
        np_attn_error=float(max(np.max(np.abs(a-b.numpy())) for a,b in zip(np_attention,attention)))
        examples=[]
        for index in range(3):
            x=images[index:index+1]
            edited=x.clone();edited[0,0,2,4]=1-edited[0,0,2,4]
            order=torch.arange(16);order[5]=6;order[6]=5
            original=model(x);pixel=model(edited)
            fixed_tokens,_=model.encode(x,order,False)
            moved_tokens,_=model.encode(x,order,True)
            fixed=model.head(fixed_tokens[:,0]);moved=model.head(moved_tokens[:,0])
            image_features,weights=model.encode(x,capture=True)
            joint_error=(moved-original).abs().max().item()
            assert joint_error<1e-4 and (fixed-original).abs().max()>1e-5
            pixel_change=(pixel-original).abs().max().item()
            assert pixel_change>1e-5 or x[0,0,2,4].item()==.5
            examples.append({'source_id':index+1,'label':int(labels[index]),'image':x[0,0].tolist(),
                             'logits':original[0].tolist(),'probabilities':original.softmax(-1)[0].tolist(),
                             'changed_pixel':[2,4],'old_pixel':x[0,0,2,4].item(),'new_pixel':edited[0,0,2,4].item(),
                             'edited_logits':pixel[0].tolist(),'edited_probabilities':pixel.softmax(-1)[0].tolist(),
                             'patch_swap':[5,6],'fixed_position_logits':fixed[0].tolist(),'joint_permutation_error':joint_error,
                             'patch_features':image_features[0,1:].tolist(),
                             'last_head0_cls_attention':weights[-1][0,0,0].tolist()})
        # A fixed training-only PCA defines one common feature coordinate/color system.
        train_features=torch.cat([model.encode(chunk)[0][:,1:] for chunk in data['train'][0].split(128)]).reshape(-1,32).numpy().astype(float)
        center=train_features.mean(0);covariance=(train_features-center).T@(train_features-center)/(len(train_features)-1)
        values,basis=np.linalg.eigh(covariance);order=np.argsort(values)[::-1];values=values[order];basis=basis[:,order[:3]]
        for column in range(3):
            if basis[np.abs(basis[:,column]).argmax(),column]<0:basis[:,column]*=-1
        training_projection=(train_features-center)@basis
        for example in examples:
            feature=np.array(example['patch_features']);projection=(feature-center)@basis
            example['patch_pca']=projection.tolist()
        fresh_image=images[2:3].clone()
        fresh_changed=fresh_image.clone();fresh_changed[0,0,2,3]=1-fresh_changed[0,0,2,3]
        fresh_logits=model(fresh_changed)
        assert (fresh_logits-model(fresh_image)).abs().max()>1e-5
        target=np.array(examples[1]['patch_features']);target=target/np.linalg.norm(target,axis=1,keepdims=True)
        query=np.array(examples[0]['patch_features'])[5];query=query/np.linalg.norm(query)
        similarity=target@query
        matches=np.argsort(-similarity,kind='stable')[:3]
        results={'numpy_logit_max_error':error,'numpy_attention_max_error':np_attn_error,
                 'reload_prediction_equality':True,'examples':examples,
                 'pca':{'training_mean':center.tolist(),'basis':basis.tolist(),'explained_variance_ratio':(values[:3]/values.sum()).tolist(),
                        'training_min':training_projection.min(0).tolist(),'training_max':training_projection.max(0).tolist()},
                 'patch_matching':{'query_source':1,'query_patch':5,'target_source':2,
                                   'top_patch_ids':matches.tolist(),'cosines':similarity[matches].tolist(),
                                   'all_cosines':similarity.tolist()},
                 'fresh_image_change':{'source_id':3,'coordinate':[2,3],
                                       'old':fresh_image[0,0,2,3].item(),'new':fresh_changed[0,0,2,3].item(),
                                       'logits':fresh_logits[0].tolist(),'probabilities':fresh_logits.softmax(-1)[0].tolist()}}
    # Gradient of displayed DINO single-row loss checked by central differences.
    student=np.array([.1,.2,-.1]);target=softmax((np.array([.4,.1,-.2])-np.array([.1,0,-.1]))/.2)
    def loss(s): return -np.sum(target*np.log(softmax(s/.5)))
    finite=[]
    for j in range(3):
        plus=student.copy();minus=student.copy();plus[j]+=1e-6;minus[j]-=1e-6
        finite.append((loss(plus)-loss(minus))/2e-6)
    analytic=(softmax(student/.5)-target)/.5
    results['dino_finite_difference_error']=float(np.max(np.abs(finite-analytic)))
    assert results['dino_finite_difference_error']<1e-8
    # Fresh independent patch-projection task; own-value-preserving null in the projection kernel.
    patch=np.array([2.,1.,4.,0.]);weight=np.array([[1,0,0,1],[0,1,-1,0]]);bias=np.array([.5,1.])
    changed=patch.copy();changed[2]=2
    null=patch+np.array([-1,0,0,1])
    results['fresh_patch']={'input':patch.tolist(),'output':(weight@patch+bias).tolist(),
                            'changed_input':changed.tolist(),'changed_output':(weight@changed+bias).tolist(),
                            'null_input':null.tolist(),'null_output':(weight@null+bias).tolist()}
    assert np.array_equal(weight@patch+bias,weight@null+bias)
    gram_features=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    gram_changed=gram_features.copy();gram_changed[3]=[1,0]
    rotation=np.array([[0.,-1.],[1.,0.]])
    gram=gram_features@gram_features.T
    results['fresh_gram']={'features':gram_features.tolist(),'gram':gram.tolist(),
                           'changed_features':gram_changed.tolist(),
                           'changed_loss':float(np.square(gram_changed@gram_changed.T-gram).sum()),
                           'rotation_loss':float(np.square((gram_features@rotation)@(gram_features@rotation).T-gram).sum())}
    with torch.no_grad():
        window=mechanisms['ShiftedWindowAttention'](1,1,2,1).double()
        for parameter in window.parameters():parameter.zero_()
        window.qkv.weight[2,0]=1;window.output.weight[0,0]=1
        grid=torch.zeros(1,6,6,1,dtype=torch.float64);grid[0,1,1,0]=10
        baseline=window(grid)[0,2,2,0].item()
        window.relative_bias[0,8]=np.log(2)
        changed=window(grid)[0,2,2,0].item()
        distant=grid.clone();distant[0,5,5,0]=30
        null=window(distant)[0,2,2,0].item()
        assert abs(baseline-2.5)<1e-12 and abs(changed-4)<1e-12 and changed==null
        results['fresh_window_bias']={'source':[1,1],'destination':[2,2],'source_value':10,
                                       'baseline':baseline,'offset_bias':[[1,1],float(np.log(2))],
                                       'changed':changed,'distant_null':null}
    first=mechanisms['mean_matrix'](6,6,2,0);shifted=mechanisms['mean_matrix'](6,6,2,1)
    source=np.zeros(36);source[0]=16
    results['fresh_window_path']={'source':[0,0],'destination':[2,2],
                                 'fixed_windows':float((first@first@source)[14]),
                                 'shifted_windows':float((shifted@first@source)[14]),
                                 'distant_source_coefficient':float((shifted@first)[14,35])}
    # Equal prototypes show a real stationary collapse, not a promised safeguard.
    equal=torch.zeros(2,1,3,dtype=torch.float64,requires_grad=True)
    teacher=torch.zeros_like(equal,requires_grad=True)
    collapse=mechanisms['dino_cross_view_loss'](equal,teacher,torch.zeros(3));collapse.backward()
    results['collapse']={'loss':collapse.item(),'student_gradient_max':equal.grad.abs().max().item(),'teacher_gradient_absent':teacher.grad is None}
    assert equal.grad.abs().max()<1e-15 and teacher.grad is None
    (ROOT/'author-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'numpy_logit_error':error,'numpy_attention_error':np_attn_error,
                     'examples':[{'source':e['source_id'],'label':e['label'],'probability':e['probabilities'][e['label']],
                                  'edited_probability':e['edited_probabilities'][e['label']],
                                  'swap_max_change':float(np.max(np.abs(np.array(e['logits'])-e['fixed_position_logits']))),
                                  'joint_error':e['joint_permutation_error']} for e in examples],
                     'matching':results['patch_matching'],'pca_variance':results['pca']['explained_variance_ratio'],
                     'fresh_patch':results['fresh_patch'],'collapse':results['collapse']},indent=2))


if __name__=='__main__':main()
