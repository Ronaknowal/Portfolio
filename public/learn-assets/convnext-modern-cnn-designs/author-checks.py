"""Independent small numerical oracles and saved-model interventions; no fitting."""
from pathlib import Path
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import erf
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("masked_reconstruction",HERE/"masked-reconstruction.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def convolution(values, weight, bias, stride=1, padding=0, depthwise=False):
    # Independent HWC cross-correlation loops, one specimen.
    padded = np.pad(values,((padding,padding),(padding,padding),(0,0)))
    outputs, _, kh, kw = weight.shape
    height = (padded.shape[0]-kh)//stride+1
    width = (padded.shape[1]-kw)//stride+1
    result = np.empty((height,width,outputs))
    for row in range(height):
        for col in range(width):
            patch = padded[row*stride:row*stride+kh,col*stride:col*stride+kw]
            for channel in range(outputs):
                local = patch[:,:,channel:channel+1] if depthwise else patch
                result[row,col,channel] = np.sum(local*weight[channel].transpose(1,2,0))+bias[channel]
    return result


def normalize(values, weight, bias):
    centered = values-values.mean(-1,keepdims=True)
    return centered / np.sqrt((centered**2).mean(-1,keepdims=True)+1e-6)*weight+bias


def global_response(values, scale, shift):
    norms = np.sqrt((values**2).sum((0,1),keepdims=True))
    relative = norms / (norms.mean(-1,keepdims=True)+1e-6)
    return values+scale*values*relative+shift


def independent_prediction(state, pixels, visible):
    weights = {key:np.asarray(value) for key,value in state.items()}
    def block(features, prefix, mask):
        features = features*mask[:,:,None]
        values = convolution(features,weights[prefix+".spatial.weight"],
                             weights[prefix+".spatial.bias"],padding=1,depthwise=True)
        values = normalize(values*mask[:,:,None],weights[prefix+".norm.weight"],
                           weights[prefix+".norm.bias"])
        values = values@weights[prefix+".expand.weight"].T+weights[prefix+".expand.bias"]
        values = .5*values*(1+erf(values/np.sqrt(2)))*mask[:,:,None]
        if prefix+".response.scale" in weights:
            values = global_response(values,weights[prefix+".response.scale"],
                                     weights[prefix+".response.shift"])
        values = values@weights[prefix+".project.weight"].T+weights[prefix+".project.bias"]
        return (features+values)*mask[:,:,None]
    pixel_mask = np.repeat(np.repeat(visible,2,axis=0),2,axis=1)
    features = convolution((pixels*pixel_mask)[:,:,None],weights["stem.weight"],
                           weights["stem.bias"],stride=2)
    features = normalize(features,weights["stem_norm.weight"],weights["stem_norm.bias"])*visible[:,:,None]
    features = block(features,"encoder.0",visible)
    features = block(features,"encoder.1",visible)
    features += weights["mask_token"][0,:,0,0]*(1-visible[:,:,None])
    features = block(features,"decoder",np.ones((4,4)))
    pixels4 = convolution(features,weights["pixel_head.weight"],weights["pixel_head.bias"])
    image = np.empty((8,8))
    for row in range(4):
        for col in range(4):
            image[2*row:2*row+2,2*col:2*col+2] = pixels4[row,col].reshape(2,2)
    return image


def main():
    result = {}
    values = np.array([[[3.,0.]],[[4.,12.]]]) # H=2,W=1,C=2; channel norms5,12.
    initial = global_response(values,np.zeros(2),np.zeros(2))
    coupled = global_response(values,np.array([.5,-.5]),np.zeros(2))
    edited = values.copy()
    edited[1,0,1] = 0
    changed = global_response(edited,np.array([.5,-.5]),np.zeros(2))
    result["grn"] = {"input":values.tolist(),"norms":[5,12],"relative":(np.array([5,12])/(8.5+1e-6)).tolist(),
        "initial_identity_error":float(np.abs(initial-values).max()),
        "output":coupled.tolist(),"edited_output":changed.tolist(),
        "zero_input_output":global_response(np.zeros_like(values),np.array([.5,-.5]),np.zeros(2)).tolist()}
    layer = experiment.GlobalResponse(2).double()
    inputs = torch.tensor(values[None],dtype=torch.float64,requires_grad=True)
    loss = layer(inputs).square().sum()/2
    loss.backward()
    result["grn"]["initial_half_squared_loss"] = float(loss.detach())
    result["grn"]["initial_scale_gradient"] = layer.scale.grad.tolist()
    result["grn"]["initial_shift_gradient"] = layer.shift.grad.tolist()
    result["grn"]["initial_input_gradient"] = inputs.grad[0].tolist()
    numeric = []
    for index in range(2):
        def objective(gamma):
            return (global_response(values,gamma,np.zeros(2))**2).sum()/2
        plus,minus=np.zeros(2),np.zeros(2)
        plus[index],minus[index]=1e-6,-1e-6
        numeric.append((objective(plus)-objective(minus))/2e-6)
    result["grn"]["scale_gradient_finite_difference_error"] = float(np.abs(numeric-layer.scale.grad.numpy()).max())
    # LN across channels at each position versus normalization over the whole specimen.
    axes = np.array([[[1.,3.],[101.,103.]]])
    local = normalize(axes,np.ones(2),np.zeros(2))
    whole = (axes-axes.mean())/np.sqrt(axes.var()+1e-6)
    result["normalization_axes"] = {"input":axes.tolist(),"channel_ln":local.tolist(),"single_group":whole.tolist()}
    three = np.array([[[1.,3.,7.],[101.,103.,107.]]])
    three_edit = three.copy()
    three_edit[0,1,0] = 105
    three_shift = three.copy()
    three_shift[0,1] += 10
    three_output = normalize(three,np.ones(3),np.zeros(3))
    result["normalization_axes"]["three_channel"] = {
        "input":three.tolist(),"original":three_output.tolist(),
        "edited":normalize(three_edit,np.ones(3),np.zeros(3)).tolist(),
        "location_shift_max_change":float(np.abs(normalize(three_shift,np.ones(3),np.zeros(3))-three_output).max())}
    # Exact inference-time folding of two linear Conv+BN branches and identity.
    image = np.arange(1.,26.).reshape(5,5,1)
    kernel = np.arange(9.).reshape(1,1,3,3)/10
    small = np.array([[[[2.]]]])
    def fuse(weight,bias,mean,variance,gamma,beta):
        scale=gamma/np.sqrt(variance+1e-5)
        return weight*scale,(bias-mean)*scale+beta
    first,first_bias=fuse(kernel,.4,1.,4.,3.,-.2)
    second,second_bias=fuse(small,-.3,-2.,1.,.5,.7)
    folded=first.copy()
    folded[:,:,1,1]+=second[:,:,0,0]+1
    separately=convolution(image,first,np.array([first_bias]),padding=1)+convolution(
        image,second,np.array([second_bias]))+image
    combined=convolution(image,folded,np.array([first_bias+second_bias]),padding=1)
    nonlinear_separate=np.maximum(convolution(image,first,np.array([first_bias]),padding=1),0)+np.maximum(
        convolution(-image,second,np.array([second_bias])),0)
    nonlinear_folded=np.maximum(convolution(image,first,np.array([first_bias]),padding=1)+
        convolution(-image,second,np.array([second_bias])),0)
    result["fusion"]={"folded_kernel":folded.tolist(),"folded_bias":float(first_bias+second_bias),
        "maximum_linear_error":float(np.abs(separately-combined).max()),
        "branch_relu_noncommutation_error":float(np.abs(nonlinear_separate-nonlinear_folded).max()),
        "center_output":float(combined[2,2,0])}
    packet=json.loads((HERE/"calculated-inputs.json").read_text())
    interventions=[]
    for run in packet["runs"]:
        if run["seed"]!=1:
            continue
        model=experiment.SmallMaskedModel(run["global_response"])
        model.load_state_dict({k:torch.tensor(v) for k,v in run["model_state"].items()})
        model.eval()
        for example in run["examples"]:
            original=np.array(example["input"])
            visible=np.array(example["visible_patches"])
            oracle=independent_prediction(run["model_state"],original,visible)
            mask=np.repeat(np.repeat(visible,2,axis=0),2,axis=1)
            hidden_coordinate=tuple(np.argwhere(mask==0)[0])
            visible_coordinate=tuple(np.argwhere(mask==1)[0])
            hidden_edit,visible_edit=original.copy(),original.copy()
            hidden_edit[hidden_coordinate]=1-original[hidden_coordinate]
            visible_edit[visible_coordinate]=1-original[visible_coordinate]
            hidden_result=independent_prediction(run["model_state"],hidden_edit,visible)
            visible_result=independent_prediction(run["model_state"],visible_edit,visible)
            swapped_mask=visible.copy()
            first_hidden_patch=tuple(np.argwhere(visible==0)[0])
            first_visible_patch=tuple(np.argwhere(visible==1)[0])
            swapped_mask[first_hidden_patch],swapped_mask[first_visible_patch]=1,0
            swapped_result=independent_prediction(run["model_state"],original,swapped_mask)
            mask_tensor=torch.tensor(visible,dtype=torch.float32)[None,None]
            with torch.no_grad():
                native=model(torch.tensor(original,dtype=torch.float32)[None,None],mask_tensor)[0,0].numpy()
            hidden=(1-mask).astype(bool)
            before=float(((oracle-original)[hidden]**2).mean())
            after=float(((hidden_result-hidden_edit)[hidden]**2).mean())
            interventions.append({"global_response":run["global_response"],"source_id":example["source_id"],
                "numpy_vs_native_max_error":float(np.abs(oracle-native).max()),
                "numpy_vs_saved_max_error":float(np.abs(oracle-example["reconstruction"]).max()),
                "hidden_coordinate":[int(v) for v in hidden_coordinate],"visible_coordinate":[int(v) for v in visible_coordinate],
                "hidden_edit_prediction_max_change":float(np.abs(hidden_result-oracle).max()),
                "visible_edit_prediction_max_change":float(np.abs(visible_result-oracle).max()),
                "hidden_target_mse_before":before,"hidden_target_mse_after":after,
                "hidden_pixel_count":int(hidden.sum()),"visible_patch_count":int(visible.sum()),
                "patch_swap_prediction_max_change":float(np.abs(swapped_result-oracle).max()),
                "swapped_visible_mask":swapped_mask.tolist(),
                "visible_edit_reconstruction":visible_result.tolist()})
    result["interventions"]=interventions
    assert result["fusion"]["maximum_linear_error"]<1e-10
    assert result["grn"]["scale_gradient_finite_difference_error"]<1e-7
    assert max(row["numpy_vs_native_max_error"] for row in interventions)<1e-5
    assert all(row["hidden_edit_prediction_max_change"]==0 for row in interventions)
    assert all(row["visible_edit_prediction_max_change"]>1e-4 for row in interventions)
    (HERE/"author-check-results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"grn":result["grn"],"fusion":result["fusion"],
        "interventions":[{k:v for k,v in row.items() if k!="visible_edit_reconstruction"} for row in interventions]}))


if __name__=="__main__":
    main()
