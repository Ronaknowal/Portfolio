"""Bounded masked-image learning with ConvNeXt-style blocks.

Actual optical-digit data are adjacent; no downloads. This small dense-masked
experiment is not a reproduction of ImageNet FCMAE or its sparse runtime.
"""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


class GlobalResponse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, features):
        norms = torch.linalg.vector_norm(features, dim=(1,2), keepdim=True)
        relative = norms / (norms.mean(-1, keepdim=True) + 1e-6)
        return features + self.scale * features * relative + self.shift


class SpatialChannelBlock(nn.Module):
    def __init__(self, channels=12, global_response=False):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4*channels)
        self.project = nn.Linear(4*channels, channels)
        self.response = GlobalResponse(4*channels) if global_response else nn.Identity()

    def forward(self, features, visible=None, expose_expansion=False):
        if visible is None:
            visible = torch.ones_like(features[:, :1])
        features = features * visible
        spatial = self.spatial(features) * visible
        channel_last = self.norm(spatial.permute(0,2,3,1))
        expanded = F.gelu(self.expand(channel_last)) * visible.permute(0,2,3,1)
        mixed = self.project(self.response(expanded))
        output = (features + mixed.permute(0,3,1,2)) * visible
        return (output, expanded) if expose_expansion else output


class SmallMaskedModel(nn.Module):
    def __init__(self, global_response=False):
        super().__init__()
        self.stem = nn.Conv2d(1,12,2,stride=2)
        self.stem_norm = nn.LayerNorm(12, eps=1e-6)
        self.encoder = nn.ModuleList([
            SpatialChannelBlock(global_response=global_response) for _ in range(2)])
        self.mask_token = nn.Parameter(torch.zeros(1,12,1,1))
        self.decoder = SpatialChannelBlock(global_response=False)
        self.pixel_head = nn.Conv2d(12,4,1)

    def encode(self, images, visible):
        pixel_visibility = visible.repeat_interleave(2,2).repeat_interleave(2,3)
        features = self.stem(images * pixel_visibility)
        features = self.stem_norm(features.permute(0,2,3,1)).permute(0,3,1,2) * visible
        for index, block in enumerate(self.encoder):
            if index == len(self.encoder)-1:
                features, expanded = block(features, visible, expose_expansion=True)
            else:
                features = block(features, visible)
        return features, expanded

    def forward(self, images, visible):
        encoded, _ = self.encode(images, visible)
        decoder_input = encoded + self.mask_token * (1-visible)
        decoded = self.decoder(decoder_input)
        return F.pixel_shuffle(self.pixel_head(decoded), 2)


def visibility(batch_size, generator):
    # Exactly ten hidden 2x2 patches out of sixteen, six visible.
    ordering = torch.rand(batch_size,16,generator=generator).argsort(1)
    mask = torch.zeros(batch_size,16)
    mask.scatter_(1,ordering[:,:6],1)
    return mask.reshape(batch_size,1,4,4)


def masked_mse(predictions, targets, visible):
    hidden_pixels = (1-visible).repeat_interleave(2,2).repeat_interleave(2,3)
    return ((predictions-targets).square()*hidden_pixels).sum()/hidden_pixels.sum()


def evaluate_reconstruction(model, images, masks):
    model.eval()
    with torch.no_grad():
        return float(torch.stack([masked_mse(model(images, mask),images,mask)
                                  for mask in masks]).mean())


def feature_matrix(model, images):
    model.eval()
    with torch.no_grad():
        visible = torch.ones(len(images),1,4,4)
        features, expanded = model.encode(images, visible)
        vectors = expanded.permute(0,3,1,2).flatten(2)
        norm = torch.linalg.vector_norm(vectors,dim=2)
        # Zero/near-zero channel vectors are reported separately, not treated as valid cosine pairs.
        valid = norm > 1e-8
        unit = vectors / norm.clamp_min(1e-8)[:,:,None]
        cosine = unit @ unit.transpose(1,2)
        pair_mask = valid[:,:,None] & valid[:,None,:]
        pair_mask &= ~torch.eye(vectors.shape[1],dtype=torch.bool)[None]
        diversity = float(((1-cosine)/2)[pair_mask].mean())
        return features.flatten(1).numpy(), {
            "mean_nonself_cosine_distance":diversity,
            "near_zero_channel_fraction":float((~valid).float().mean()),
            "threshold":1e-8,
            "measurement":"clean-image final encoder expansion, before GRN; per-image channel spatial vectors",
        }


def probe_scores(train_features, train_labels, dev_features, dev_labels):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1,max_iter=2000))
    probe.fit(train_features,train_labels)
    probabilities = probe.predict_proba(dev_features)
    return {
        "training_correct":int((probe.predict(train_features)==train_labels).sum()),
        "development_correct":int((probabilities.argmax(1)==dev_labels).sum()),
        "development_count":len(dev_labels),
        "development_cross_entropy":float(log_loss(dev_labels,probabilities,labels=np.arange(10))),
        "development_predictions":probabilities.argmax(1).tolist(),
    }


def main():
    records = np.genfromtxt(HERE/"digits-400.csv", delimiter=",", names=True)
    pixels = np.column_stack([records[f"pixel_{i}"] for i in range(64)])
    source_ids = records["source_id"].astype(int)
    labels = records["digit"].astype(int)
    if len(np.unique(pixels,axis=0)) != 400 or len(np.unique(source_ids)) != 400:
        raise ValueError("revise duplicate grouping before splitting")
    train, dev = train_test_split(np.arange(400),test_size=.3,random_state=22,stratify=labels)
    images = torch.tensor(pixels/16,dtype=torch.float32).reshape(400,1,8,8)
    eval_generator = torch.Generator().manual_seed(999)
    evaluation_masks = [visibility(400,eval_generator) for _ in range(4)]
    training_masks = [mask[train] for mask in evaluation_masks]
    development_masks = [mask[dev] for mask in evaluation_masks]
    mean_image = images[train].mean(0,keepdim=True)
    mean_image_mse = float(torch.stack([
        masked_mse(mean_image.expand(len(dev),-1,-1,-1),images[dev],mask)
        for mask in development_masks]).mean())
    result = {
        "environment":{"python":platform.python_version(),"numpy":np.__version__,
                       "torch":torch.__version__,"sklearn":sklearn.__version__,"threads":1},
        "data_audit":{"rows":400,"unique_images":400,"unique_source_ids":400},
        "training_source_ids":source_ids[train].tolist(),
        "development_source_ids":source_ids[dev].tolist(),
        "development_labels":labels[dev].tolist(),
        "baseline_training_mean_image":mean_image[0,0].tolist(),
        "baseline_masked_development_mse":mean_image_mse,
        "raw_pixel_linear_probe":probe_scores(pixels[train]/16,labels[train],pixels[dev]/16,labels[dev]),
        "runs":[],
    }
    for seed in (1,2,3):
        for use_grn in (False,True):
            torch.manual_seed(seed)
            model = SmallMaskedModel(use_grn)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            mask_generator = torch.Generator().manual_seed(100+seed)
            trace = []
            for step in range(601):
                if step in (0,1,100,300,600):
                    trace.append({"step":step,
                        "training_masked_mse":evaluate_reconstruction(model,images[train],training_masks),
                        "development_masked_mse":evaluate_reconstruction(model,images[dev],development_masks)})
                if step == 600:
                    break
                model.train()
                visible = visibility(len(train),mask_generator)
                optimizer.zero_grad()
                loss = masked_mse(model(images[train],visible),images[train],visible)
                loss.backward()
                optimizer.step()
            train_features, train_diagnostics = feature_matrix(model,images[train])
            dev_features, dev_diagnostics = feature_matrix(model,images[dev])
            probe = probe_scores(train_features,labels[train],dev_features,labels[dev])
            run = {"seed":seed,"global_response":use_grn,
                   "parameters":sum(p.numel() for p in model.parameters()),
                   "trace":trace,"linear_probe":probe,
                   "training_feature_diagnostics":train_diagnostics,
                   "development_feature_diagnostics":dev_diagnostics}
            if seed == 1:
                model.eval()
                selected = [0,1]
                run["examples"] = []
                for index in selected:
                    mask = development_masks[0][index:index+1]
                    original = images[dev[index]:dev[index]+1]
                    with torch.no_grad():
                        prediction = model(original,mask)
                    run["examples"].append({
                        "source_id":int(source_ids[dev[index]]),"actual":int(labels[dev[index]]),
                        "input":original[0,0].tolist(),"visible_patches":mask[0,0].tolist(),
                        "reconstruction":prediction[0,0].tolist(),
                        "masked_mse":float(masked_mse(prediction,original,mask))})
                run["model_state"] = {key:value.detach().tolist() for key,value in model.state_dict().items()}
            result["runs"].append(run)
            print(seed,use_grn,trace[-1]["development_masked_mse"],
                  probe["development_correct"],dev_diagnostics["mean_nonself_cosine_distance"],flush=True)
    (HERE/"calculated-inputs.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
