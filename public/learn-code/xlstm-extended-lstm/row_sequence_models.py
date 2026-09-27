"""Train small recurrent digit readers, offline, with explicit train/validation/test roles.

Requirements: numpy, torch. Keep optdigits.tra, optdigits.tes and optdigits.names beside
this file; run python row_sequence_models.py. No pretrained model or network needed.
These one-head instructional blocks are not published xLSTM-7B checkpoints.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

ROOT = Path(__file__).parent


def load_data():
    training = np.loadtxt(ROOT/"optdigits.tra",delimiter=",",dtype=np.int64)
    testing = np.loadtxt(ROOT/"optdigits.tes",delimiter=",",dtype=np.int64)
    assert len(np.unique(np.concatenate((training[:,:64],testing[:,:64])),axis=0)) == 5620
    random = np.random.default_rng(157)
    ids = {"fit":[],"validation":[]}
    for label in range(10):
        indices = random.permutation(np.flatnonzero(training[:,-1]==label))
        ids["fit"].extend(indices[:100].tolist())
        ids["validation"].extend(indices[100:130].tolist())
    roles = {}
    for name,indices in ids.items():
        rows = training[indices]
        roles[name] = (torch.tensor(rows[:,:64]/16,dtype=torch.float32).reshape(-1,8,8),
                       torch.tensor(rows[:,-1],dtype=torch.long))
    roles["test"] = (torch.tensor(testing[:,:64]/16,dtype=torch.float32).reshape(-1,8,8),
                     torch.tensor(testing[:,-1],dtype=torch.long))
    metadata = {"split_seed":157,"training_source_ids":{k:[i+1 for i in v] for k,v in ids.items()},
                "test_source_ids":list(range(1,1798)),"unique_feature_rows":5620,
                "hashes":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ["optdigits.tra","optdigits.tes","optdigits.names"]}}
    return roles,metadata


class ScalarMemory(nn.Module):
    """One-head sLSTM: exponential write, sigmoid retention, tanh content."""
    def __init__(self,width=16):
        super().__init__()
        self.width = width
        self.input_gates = nn.Linear(width,4*width)
        self.recurrent_gates = nn.Linear(width,4*width,bias=False)
        with torch.no_grad():
            self.input_gates.bias[width:2*width].fill_(1.)

    def forward(self,sequence,state=None):
        if state is None:
            zero=sequence.new_zeros(sequence.shape[0],self.width)
            state=(zero,zero.clone(),zero.clone(),zero.clone())
        hidden,cell,normalizer,log_scale=state
        outputs=[]
        for token in sequence.unbind(dim=1):
            write_log,forget_raw,output_raw,content_raw=(
                self.input_gates(token)+self.recurrent_gates(hidden)).chunk(4,dim=-1)
            forget_log=functional.logsigmoid(forget_raw)
            new_scale=torch.maximum(forget_log+log_scale,write_log)
            write=torch.exp(write_log-new_scale)
            retain=torch.exp(forget_log+log_scale-new_scale)
            cell=retain*cell+write*content_raw.tanh()
            normalizer=retain*normalizer+write
            hidden=output_raw.sigmoid()*(cell/normalizer)
            log_scale=new_scale
            outputs.append(hidden)
        return torch.stack(outputs,dim=1),(hidden,cell,normalizer,log_scale)


class MatrixMemory(nn.Module):
    """One mLSTM head; C has key rows and value columns, with the correct scale floor."""
    def __init__(self,width=16,key_width=8):
        super().__init__()
        self.width=width
        self.key_width=key_width
        self.queries=nn.Linear(width,key_width)
        self.keys=nn.Linear(width,key_width)
        self.values=nn.Linear(width,width)
        self.gates=nn.Linear(width,2)
        self.output_gate=nn.Linear(width,width)
        self.read_norm=nn.RMSNorm(width,eps=1e-6)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([-1.,1.]))

    def forward(self,sequence,state=None):
        if state is None:
            batch=sequence.shape[0]
            state=(sequence.new_zeros(batch,self.key_width,self.width),
                   sequence.new_zeros(batch,self.key_width),sequence.new_zeros(batch))
        cell,normalizer,log_scale=state
        queries=self.queries(sequence)/(self.key_width**.5)
        keys=self.keys(sequence)
        values=self.values(sequence)
        gate_raw=15*torch.tanh(self.gates(sequence)/15)
        output_gate=self.output_gate(sequence).sigmoid()
        outputs=[]
        for index in range(sequence.shape[1]):
            write_log=gate_raw[:,index,0]
            forget_log=functional.logsigmoid(gate_raw[:,index,1])
            new_scale=torch.maximum(forget_log+log_scale,write_log)
            write=torch.exp(write_log-new_scale)
            retain=torch.exp(forget_log+log_scale-new_scale)
            key,value,query=keys[:,index],values[:,index],queries[:,index]
            cell=retain[:,None,None]*cell+write[:,None,None]*key[:,:,None]*value[:,None,:]
            normalizer=retain[:,None]*normalizer+write[:,None]*key
            numerator=torch.einsum("bkv,bk->bv",cell,query)
            denominator=torch.maximum((normalizer*query).sum(-1).abs(),torch.exp(-new_scale))
            read=numerator/denominator[:,None]
            outputs.append(output_gate[:,index]*self.read_norm(read))
            log_scale=new_scale
        return torch.stack(outputs,dim=1),(cell,normalizer,log_scale)


class DigitReader(nn.Module):
    def __init__(self,kind,width=16):
        super().__init__()
        self.kind=kind
        self.input_projection=nn.Linear(8,width)
        self.pre_norm=nn.RMSNorm(width,eps=1e-6)
        self.sequence_model=(nn.LSTM(width,width,batch_first=True) if kind=="lstm" else
                             ScalarMemory(width) if kind=="slstm" else MatrixMemory(width))
        self.post_norm=nn.RMSNorm(width,eps=1e-6)
        self.expand=nn.Linear(width,2*width)
        self.gate=nn.Linear(width,2*width)
        self.contract=nn.Linear(2*width,width)
        self.classifier=nn.Linear(width,10)

    def forward(self,pixel_rows,state=None):
        embedded=self.input_projection(pixel_rows)
        mixed,state=self.sequence_model(self.pre_norm(embedded),state)
        residual=embedded+mixed
        normalized=self.post_norm(residual)
        output=residual+self.contract(functional.silu(self.expand(normalized))*self.gate(normalized))
        return self.classifier(output),state


def metrics(logits,labels):
    predictions=logits.argmax(dim=-1)
    confusion=torch.zeros(10,10,dtype=torch.long)
    for truth,prediction in zip(labels,predictions):
        confusion[truth,prediction]+=1
    return {"count":len(labels),"errors":int((predictions!=labels).sum()),
            "cross_entropy":float(functional.cross_entropy(logits,labels)),"confusion":confusion.tolist()}


def main():
    torch.set_num_threads(1)
    roles,metadata=load_data()
    results={"data":metadata,"protocol":{"epochs":150,"learning_rate":.003,"fit_rows":1000,
             "validation_rows":300,"test_rows":1797,"seeds":[19,43]},"models":[]}
    arrays={"validation_images":roles["validation"][0].numpy(),
            "validation_labels":roles["validation"][1].numpy()}
    for kind in ["lstm","slstm","mlstm"]:
        for seed in [19,43]:
            torch.manual_seed(seed)
            model=DigitReader(kind)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            best_loss=float("inf")
            curve=[]
            for epoch in range(1,151):
                model.train()
                optimizer.zero_grad()
                logits,_=model(roles["fit"][0])
                loss=functional.cross_entropy(logits[:,-1],roles["fit"][1])
                loss.backward()
                gradient_norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(gradient_norm)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_logits,_=model(roles["validation"][0])
                    validation_loss=float(functional.cross_entropy(validation_logits[:,-1],roles["validation"][1]))
                curve.append({"epoch":epoch,"fit_loss_before_update":float(loss.detach()),
                              "validation_loss_after_update":validation_loss,"gradient_norm":float(gradient_norm)})
                if validation_loss<best_loss:
                    best_loss=validation_loss
                    best_epoch=epoch
                    selected=copy.deepcopy(model.state_dict())
            model.load_state_dict(selected)
            prefix=f"{kind}_seed{seed}"
            report={"kind":kind,"seed":seed,"selected_epoch":best_epoch,
                    "parameters":sum(p.numel() for p in model.parameters()),"training_curve":curve}
            with torch.no_grad():
                for role,(images,labels) in roles.items():
                    for condition,inputs in [("clean",images),("reversed",images.flip(1))]:
                        logits,_=model(inputs)
                        report[f"{role}_{condition}"]=metrics(logits[:,-1],labels)
                        if role=="validation": arrays[f"{prefix}_{condition}_logits"]=logits.numpy()
                images=roles["validation"][0][:8]
                whole,whole_state=model(images)
                left,left_state=model(images[:,:3])
                right,right_state=model(images[:,3:],left_state)
                reset_right,_=model(images[:,3:])
                report["split_carry_maximum_logit_error"]=float((whole-torch.cat((left,right),dim=1)).abs().max())
                report["reset_boundary_maximum_final_logit_difference"]=float((whole[:,-1]-reset_right[:,-1]).abs().max())
                changed=images.clone();changed[:,5:]=0.
                changed_logits,_=model(changed)
                report["future_edit_earlier_logit_error"]=float((whole[:,:5]-changed_logits[:,:5]).abs().max())
                assert report["split_carry_maximum_logit_error"]<1e-5
                assert report["future_edit_earlier_logit_error"]<1e-5
            for name,value in selected.items(): arrays[f"{prefix}__{name}"]=value.numpy()
            results["models"].append(report)
            print(kind,seed,best_epoch,report["parameters"],report["validation_clean"]["errors"],
                  report["test_clean"]["errors"],report["test_reversed"]["errors"],flush=True)
    (ROOT/"row-sequence-results.json").write_text(json.dumps(results,indent=2)+"\n")
    np.savez_compressed(ROOT/"row-sequence-fits.npz",**arrays)


if __name__=="__main__":
    main()
