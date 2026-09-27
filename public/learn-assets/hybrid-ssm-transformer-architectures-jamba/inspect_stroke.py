"""Inspect saved model predictions and deliberately damage a request's caches.

Example: python inspect_stroke.py --row 1 --break-at 3 --mode recurrent-reset
Rows refer to 1-based original pendigits.tra lines; assessment labels are not tuned here.
"""
from pathlib import Path
import argparse, copy, json
import numpy as np
import torch
from stroke_models import HERE, load_fit

def run_trace(coordinates,key='MAM-37',break_at=3,mode='carry'):
    model=load_fit(key)
    points=torch.tensor(np.asarray(coordinates)/50-1,dtype=torch.float32)[None,:,:]
    with torch.no_grad():
        full=model(points,True)
        if break_at==0:
            left=full[:,:0];cache=[None]*len(model.layers)
        else:left,cache=model.stream(points[:,:break_at])
        snapshots=[]
        for kind,state in zip(model.pattern,cache):
            if state is None:snapshots.append({'kind':kind,'empty':True});continue
            snapshots.append({'kind':kind,'shapes':[list(v.shape) for v in state],
                              'norms':[float(v.norm()) for v in state]})
        if mode in ['recurrent-reset','kv-reset','convolution-reset']:
            for index,kind in enumerate(model.pattern):
                if mode=='recurrent-reset' and kind=='M':cache[index]=None
                if mode=='kv-reset' and kind=='A':cache[index]=None
                if mode=='convolution-reset' and kind=='M' and cache[index] is not None:
                    cache[index]=(cache[index][0],torch.zeros_like(cache[index][1]))
        if break_at<8:
            offset=0 if mode=='position-reset' else break_at
            right,_=model.stream(points[:,break_at:],cache,offset=offset)
            branch=torch.cat([left,right],1)
        else:branch=left
        return {'key':key,'coordinates':np.asarray(coordinates).tolist(),'break_at':break_at,
                'mode':mode,'cache_at_boundary':snapshots,
                'full_logits':full[0].tolist(),'branch_logits':branch[0].tolist(),
                'full_probabilities':full[0,-1].softmax(-1).tolist(),
                'branch_probabilities':branch[0,-1].softmax(-1).tolist(),
                'full_prediction':int(full[0,-1].argmax()),
                'branch_prediction':int(branch[0,-1].argmax()),
                'maximum_logit_difference':float((full-branch).abs().max())}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--row',type=int,default=1)
    parser.add_argument('--break-at',type=int,default=3)
    parser.add_argument('--mode',choices=['carry','recurrent-reset','kv-reset','convolution-reset','position-reset'],default='carry')
    parser.add_argument('--key',choices=['MMM-37','AAA-37','MAM-37','AMM-37','MAM-73'],default='MAM-37')
    args=parser.parse_args()
    if not 1<=args.row<=7494 or not 0<=args.break_at<=8:parser.error('Row or boundary out of range.')
    row=np.loadtxt(HERE/'pendigits.tra',delimiter=',')[args.row-1]
    print(json.dumps(run_trace(row[:16].reshape(8,2),args.key,args.break_at,args.mode),indent=2))
