"""Train a small patch Transformer with dense or sparse expert FFNs."""
from pathlib import Path
import csv, copy, hashlib, json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)

class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(16, width, bias=False)
        self.value = nn.Linear(16, width, bias=False)
        self.down = nn.Linear(width, 16, bias=False)

    def forward(self, inputs):
        return self.down(F.silu(self.gate(inputs)) * self.value(inputs))

class DigitTransformer(nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.sparse = sparse
        self.project = nn.Linear(4, 16)
        self.position = nn.Parameter(torch.randn(16, 16) * .02)
        self.norm_attention = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48, bias=False)
        self.attention_output = nn.Linear(16, 16, bias=False)
        self.norm_ffn = nn.LayerNorm(16)
        if sparse:
            self.router = nn.Linear(16, 4, bias=False)
            self.experts = nn.ModuleList([SwiGLU(16) for _ in range(4)])
        else:
            self.ffn = SwiGLU(32)
        self.norm_final = nn.LayerNorm(16)
        self.classifier = nn.Linear(16, 10)

    def forward(self, pixels, temperature=1., disabled_expert=None, trace=False):
        batch = pixels.shape[0]
        patches = pixels.reshape(batch, 4, 2, 4, 2).permute(0,1,3,2,4).reshape(batch,16,4)
        hidden = self.project(patches) + self.position
        qkv = self.qkv(self.norm_attention(hidden)).reshape(batch,16,3,2,8)
        query, key, value = qkv.permute(2,0,3,1,4).unbind(0)
        attention = (query @ key.transpose(-1,-2) / np.sqrt(8)).softmax(-1)
        context = (attention @ value).transpose(1,2).reshape(batch,16,16)
        hidden = hidden + self.attention_output(context)
        expert_inputs = self.norm_ffn(hidden).reshape(-1,16)
        auxiliary = hidden.new_tensor(0.)
        details = {}
        if self.sparse:
            scores = self.router(expert_inputs) / temperature
            probabilities = scores.softmax(-1)
            selected_scores, selected = scores.topk(2, dim=-1)
            weights = selected_scores.softmax(-1)
            combined = torch.zeros_like(expert_inputs)
            for expert_id, expert in enumerate(self.experts):
                token_ids, slots = torch.where(selected == expert_id)
                if token_ids.numel() and expert_id != disabled_expert:
                    outputs = expert(expert_inputs[token_ids])
                    combined.index_add_(0, token_ids, outputs * weights[token_ids, slots, None])
            fractions = torch.bincount(selected.flatten(), minlength=4).float() / selected.numel()
            auxiliary = 4 * (fractions.detach() * probabilities.mean(0)).sum()
            details = dict(selected=selected.reshape(batch,16,2), weights=weights.reshape(batch,16,2),
                           probabilities=probabilities.reshape(batch,16,4), counts=torch.bincount(selected.flatten(),minlength=4))
        else:
            combined = self.ffn(expert_inputs)
        hidden = hidden + combined.reshape(batch,16,16)
        logits = self.classifier(self.norm_final(hidden).mean(1))
        if trace:
            details.update(attention=attention, expert_inputs=expert_inputs.reshape(batch,16,16),
                           combined=combined.reshape(batch,16,16), logits=logits)
            return logits, auxiliary, details
        return logits, auxiliary

def load_data():
    rows = list(csv.DictReader((HERE / "optical-digits.csv").open()))
    pixels = torch.tensor([[float(r[f"pixel_{j}"]) / 16 for j in range(64)] for r in rows])
    labels = torch.tensor([int(r["label"]) for r in rows])
    roles = {role: torch.tensor([i for i,r in enumerate(rows) if r["role"] == role])
             for role in ("fit","validation","assessment")}
    return rows, pixels, labels, roles

def evaluate(model, pixels, labels):
    with torch.no_grad():
        logits, auxiliary, details = model(pixels, trace=True)
        predicted = logits.argmax(-1)
        confusion = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10,10)
        result = dict(cross_entropy=F.cross_entropy(logits,labels).item(),
                      correct=(predicted==labels).sum().item(), count=len(labels),
                      confusion=confusion.tolist(), auxiliary=auxiliary.item())
        if model.sparse:
            result["route_counts"] = details["counts"].tolist()
            result["class_route_counts"] = [torch.bincount(details["selected"][labels==c].flatten(),minlength=4).tolist()
                                           for c in range(10)]
    return result

def run_study():
    rows, pixels, labels, roles = load_data()
    fits, results = {}, []
    for condition, coefficient in (("dense",0.),("moe_0",0.),("moe_001",.01),("moe_01",.1)):
        for seed in (17,41,73):
            torch.manual_seed(seed)
            model = DigitTransformer(condition != "dense")
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
            generator = torch.Generator().manual_seed(10000+seed)
            curve, best_loss, selected_step = [], float("inf"), 0
            for step in range(1,241):
                model.train()
                sample = roles["fit"][torch.randint(len(roles["fit"]),(64,),generator=generator)]
                optimizer.zero_grad()
                logits, balance = model(pixels[sample])
                task_loss = F.cross_entropy(logits,labels[sample])
                (task_loss+coefficient*balance).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step == 1 or step % 20 == 0:
                    model.eval()
                    validation = evaluate(model,pixels[roles["validation"]],labels[roles["validation"]])
                    curve.append(dict(step=step, minibatch_ce_before=task_loss.item(),
                                      minibatch_aux_before=balance.item(), validation_ce_after=validation["cross_entropy"]))
                    if validation["cross_entropy"] < best_loss:
                        best_loss = validation["cross_entropy"]
                        selected_step, best = step, copy.deepcopy(model.state_dict())
            model.load_state_dict(best)
            model.eval()
            key = f"{condition}_{seed}"
            fits[key] = {name: values.tolist() for name,values in best.items()}
            measurements = {role:evaluate(model,pixels[index],labels[index]) for role,index in roles.items()}
            changed = pixels[roles["assessment"]].clone().reshape(-1,8,8)
            changed[:,4:,:] = 0
            measurements["assessment_lower_half_zero"] = evaluate(model,changed.reshape(-1,64),labels[roles["assessment"]])
            results.append(dict(key=key,condition=condition,seed=seed,coefficient=coefficient,
                                parameters=sum(p.numel() for p in model.parameters()),selected_step=selected_step,
                                curve=curve,measurements=measurements))
            print(key, selected_step, measurements["assessment"]["correct"], flush=True)
    (HERE/"fitted-models.json").write_text(json.dumps(fits,separators=(",",":"))+"\n")
    output=dict(python="3.12.14",numpy=np.__version__,torch=torch.__version__,
                data_sha256=hashlib.sha256((HERE/"optical-digits.csv").read_bytes()).hexdigest(),runs=results)
    (HERE/"study-results.json").write_text(json.dumps(output,indent=2)+"\n")

if __name__ == "__main__":
    run_study()
