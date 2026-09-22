"""Complete CPU convolution lesson. Inputs: adjacent attributed digits-400.csv."""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


def direct_conv2d(images, kernels, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """NCHW cross-correlation, integer symmetric geometry, explicit channel groups."""
    if images.ndim != 4 or kernels.ndim != 4:
        raise ValueError("images and kernels must have four axes")
    if min(stride, dilation, groups) < 1 or padding < 0:
        raise ValueError("invalid geometry")
    batch, in_channels, height, width = images.shape
    out_channels, per_group, kh, kw = kernels.shape
    if in_channels % groups or out_channels % groups or per_group != in_channels // groups:
        raise ValueError("channel counts must agree with groups")
    if bias is not None:
        bias = np.asarray(bias)
        if bias.shape != (out_channels,):
            raise ValueError("bias must have one value per output channel")
    oh = (height + 2*padding - dilation*(kh-1) - 1)//stride + 1
    ow = (width + 2*padding - dilation*(kw-1) - 1)//stride + 1
    if min(oh, ow) <= 0:
        raise ValueError("kernel does not fit the padded input")
    padded = np.pad(images, ((0,0),(0,0),(padding,padding),(padding,padding)))
    result_dtype = np.result_type(images,kernels) if bias is None else np.result_type(images,kernels,bias)
    result = np.zeros((batch,out_channels,oh,ow),dtype=result_dtype)
    for n in range(batch):
        for oc in range(out_channels):
            group = oc // (out_channels // groups)
            for row in range(oh):
                for col in range(ow):
                    value = 0.0 if bias is None else bias[oc]
                    for ic in range(per_group):
                        for kr in range(kh):
                            for kc in range(kw):
                                value += kernels[oc,ic,kr,kc] * padded[
                                    n, group*per_group+ic,
                                    row*stride+kr*dilation, col*stride+kc*dilation]
                    result[n,oc,row,col] = value
    return result


def receptive_trace(layers):
    size, jump, center = 1, 1, .5
    records = []
    for name, kernel, stride, left_pad, dilation in layers:
        center += (dilation*(kernel-1)/2-left_pad)*jump
        size += dilation*(kernel-1)*jump
        jump *= stride
        records.append(dict(name=name, size=size, jump=jump, first_center=center))
    return records


def fixtures():
    dtype = torch.float64
    image = torch.tensor([[1.,2.,0.],[0.,1.,3.],[2.,1.,0.]],dtype=dtype)[None,None]
    kernel = torch.tensor([[1.,-1.],[0.,1.]],dtype=dtype)[None,None]
    patches = F.unfold(image,2)
    columns = kernel.flatten(1) @ patches
    counts = F.fold(F.unfold(torch.ones_like(image),2),(3,3),2)
    folded = F.fold(patches,(3,3),2)
    assert torch.equal(folded/counts,image)
    assert torch.equal(columns.reshape(1,1,2,2),F.conv2d(image,kernel))
    generator = np.random.default_rng(8)
    parity = []
    for label, channels, outputs, kh, kw, stride, pad, dilation, groups in [
        ("ordinary",2,3,2,3,1,0,1,1),
        ("strided",2,4,3,3,2,1,1,1),
        ("dilated",2,4,3,3,1,2,2,1),
        ("grouped",4,6,2,3,1,0,1,2),
        ("depthwise_multiplier_two",2,4,3,3,1,1,1,2)]:
        x = generator.normal(size=(2,channels,5,6))
        w = generator.normal(size=(outputs,channels//groups,kh,kw))
        b = generator.normal(size=outputs)
        manual = direct_conv2d(x,w,b,stride,pad,dilation,groups)
        reference = F.conv2d(torch.tensor(x),torch.tensor(w),torch.tensor(b),
                             stride,pad,dilation,groups).numpy()
        error = float(np.max(np.abs(manual-reference)))
        assert error < 1e-12
        parity.append(dict(case=label, shape=list(manual.shape), max_absolute_error=error))
    x = torch.tensor([1.,3.,2.],dtype=dtype,requires_grad=True)
    w = torch.tensor([1.,-1.],dtype=dtype,requires_grad=True)
    y = F.conv1d(x[None,None],w[None,None]).flatten()
    loss = .5*y.square().sum()
    loss.backward()
    updated = w.detach()-.1*w.grad
    after = F.conv1d(x.detach()[None,None],updated[None,None]).flatten()
    shared_update = dict(input=x.detach().tolist(),weight=w.detach().tolist(),
                         output=y.detach().tolist(),loss=loss.item(),
                         weight_gradient=w.grad.tolist(),input_gradient=x.grad.tolist(),
                         updated_weight=updated.tolist(),updated_output=after.tolist(),
                         updated_loss=(.5*after.square().sum()).item())
    matrix = torch.tensor([[1.,-1.,0.],[0.,1.,-1.]],dtype=dtype)
    dual = torch.tensor([2.,-3.],dtype=dtype)
    adjoint = dict(matrix=matrix.tolist(),input=x.detach().tolist(),dual=dual.tolist(),
                   forward=(matrix@x.detach()).tolist(),
                   transpose=(matrix.T@dual).tolist(),
                   transpose_forward=(matrix.T@matrix@x.detach()).tolist(),
                   left_inner_product=((matrix@x.detach())@dual).item(),
                   right_inner_product=(x.detach()@(matrix.T@dual)).item())
    pool_input = torch.tensor([1.,4.,3.],dtype=dtype,requires_grad=True)
    pool_output = F.max_pool1d(pool_input[None,None],2,1)
    pool_output.sum().backward()
    tie = torch.tensor([2.,2.,1.],dtype=dtype,requires_grad=True)
    tie_output = F.max_pool1d(tie[None,None],2,1)
    tie_output.sum().backward()
    negative = torch.tensor([-2.,-3.],dtype=dtype)[None,None]
    adaptive_input = torch.arange(1.,6.,dtype=dtype,requires_grad=True)
    adaptive = F.adaptive_avg_pool1d(adaptive_input[None,None],3)
    adaptive.sum().backward()
    pooling = dict(overlap_input=pool_input.detach().tolist(),
                   overlap_output=pool_output.flatten().tolist(),
                   overlap_gradient=pool_input.grad.tolist(),
                   tie_output=tie_output.flatten().tolist(),tie_gradient=tie.grad.tolist(),
                   negative_padding=F.max_pool1d(negative,3,1,1).flatten().tolist(),
                   average_include_pad=F.avg_pool1d(negative,3,1,1,True,True).flatten().tolist(),
                   average_exclude_pad=F.avg_pool1d(negative,3,1,1,True,False).flatten().tolist(),
                   adaptive_output=adaptive.flatten().tolist(),
                   adaptive_gradient=adaptive_input.grad.tolist())
    sequence = torch.tensor([1.,0.,0.,0.,0.,0.],dtype=dtype)[None,None]
    derivative = torch.tensor([1.,0.,-1.],dtype=dtype)[None,None]
    circular = lambda z: F.conv1d(F.pad(z,(1,1),mode="circular"),derivative)
    zeros = lambda z: F.conv1d(z,derivative,padding=1)
    shift = lambda z: torch.roll(z,1,-1)
    checker = torch.tensor([1.,-1.,1.,-1.,1.,-1.],dtype=dtype)[None,None]
    lowpass = lambda z: F.conv1d(F.pad(z,(0,1),mode="circular"),
                               torch.ones(1,1,2,dtype=dtype)/2)
    translation = dict(
        circular_error=(circular(shift(sequence))-shift(circular(sequence))).abs().max().item(),
        zero_boundary_error=(zeros(shift(sequence))-shift(zeros(sequence))).abs().max().item(),
        stride_even=checker.flatten()[::2].tolist(),
        stride_after_shift=shift(checker).flatten()[::2].tolist(),
        lowpass_then_stride=lowpass(checker).flatten()[::2].tolist(),
        lowpass_shift_then_stride=lowpass(shift(checker)).flatten()[::2].tolist(),
        pool_original=F.max_pool1d(torch.tensor([0.,1.,0.,0.])[None,None],2).flatten().tolist(),
        pool_cross_boundary=F.max_pool1d(torch.tensor([0.,0.,1.,0.])[None,None],2).flatten().tolist())
    trace = receptive_trace([("conv1",3,1,1,1),("pool1",2,2,0,1),
                             ("conv2",3,1,1,1),("pool2",2,2,0,1),
                             ("conv3",3,1,1,1),("global_average_on_8_positions",8,1,0,1)])
    profiles = []
    for depth in (2,3,5,10,20):
        profile = np.array([1.])
        for _ in range(depth):
            profile = np.convolve(profile,np.ones(3)/3)
        coordinates = np.arange(-depth,depth+1)
        selected = coordinates[profile >= .01*profile.max()]
        profiles.append(dict(depth=depth,coordinates=coordinates.tolist(),
                             gradient=profile.tolist(),sum=float(profile.sum()),
                             variance=float((profile*coordinates**2).sum()),
                             support_width=2*depth+1,
                             one_percent_peak_width=int(selected[-1]-selected[0]+1)))
    hole_offsets = sorted(set(a+b for a in (-2,0,2) for b in (-2,0,2)))
    filled_offsets = sorted(set(a+b for a in (-1,0,1) for b in (-2,0,2)))
    transposed = []
    for k in (3,4):
        values = F.conv_transpose1d(torch.ones(1,1,3,dtype=dtype),
                                   torch.ones(1,1,k,dtype=dtype),stride=2)
        transposed.append(dict(kernel=k,stride=2,coverage=values.flatten().tolist()))
    torch.manual_seed(12)
    conv = nn.Conv2d(2,3,3,padding=1).double().eval()
    bn = nn.BatchNorm2d(3).double().eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([1.,-1.,.5],dtype=dtype))
        bn.running_var.copy_(torch.tensor([4.,1.,.25],dtype=dtype))
        bn.weight.copy_(torch.tensor([2.,1.,-.5],dtype=dtype))
        bn.bias.copy_(torch.tensor([.1,.2,.3],dtype=dtype))
        probe = torch.randn(2,2,4,4,dtype=dtype)
        scale = bn.weight/torch.sqrt(bn.running_var+bn.eps)
        fused_w = conv.weight*scale[:,None,None,None]
        fused_b = bn.bias+(conv.bias-bn.running_mean)*scale
        fold_error = (bn(conv(probe))-F.conv2d(probe,fused_w,fused_b,padding=1)).abs().max().item()
    laplace = np.array([[0,1,0],[1,-4,1],[0,1,0.]])
    temperature = np.array([[20,20,20],[20,30,20],[20,20,20.]])
    shape_cases = []
    for n,k,s,left,right,d in [(8,3,1,1,1,1),(8,3,2,1,1,1),
                               (8,4,1,1,2,1),(8,4,1,2,1,1),
                               (8,3,1,2,2,2),(7,3,2,0,0,1)]:
        actual = F.conv1d(F.pad(torch.ones(1,1,n,dtype=dtype),(left,right)),
                          torch.ones(1,1,k,dtype=dtype),stride=s,dilation=d)
        shape_cases.append(dict(input_size=n,kernel=k,stride=s,left=left,right=right,
                                dilation=d,output_size=actual.shape[-1],
                                first_center=.5+d*(k-1)/2-left))
    constant = torch.ones(1,1,5,dtype=dtype)
    dead_input = -torch.ones(1,1,5,dtype=dtype,requires_grad=True)
    dead_input.retain_grad()
    dead_output = F.conv1d(F.relu(dead_input),torch.ones(1,1,3,dtype=dtype)).sum()
    dead_output.backward()
    nulls = dict(zero_kernel=F.conv2d(image,torch.zeros_like(kernel))[0,0].tolist(),
                 constant_derivative=F.conv1d(constant,derivative).flatten().tolist(),
                 dead_relu_input_gradient=dead_input.grad.flatten().tolist(),
                 permuted_global_mean_error=(image.mean()-image.flatten().flip(0).mean()).abs().item())
    changed_updates = []
    for target,rate in [([0.,2.],.1),([0.,2.],.01),([-2.,1.],.1),([0.,0.],0.)]:
        changed_w = torch.tensor([1.,-1.],dtype=dtype,requires_grad=True)
        changed_y = F.conv1d(x.detach()[None,None],changed_w[None,None]).flatten()
        target_tensor = torch.tensor(target,dtype=dtype)
        changed_loss = .5*(changed_y-target_tensor).square().sum()
        gradient = torch.autograd.grad(changed_loss,changed_w)[0]
        updated_w = changed_w.detach()-rate*gradient
        updated_y = F.conv1d(x.detach()[None,None],updated_w[None,None]).flatten()
        changed_updates.append(dict(target=target,rate=rate,loss=changed_loss.item(),
                                     gradient=gradient.tolist(),weight=updated_w.tolist(),
                                     output=updated_y.tolist(),
                                     updated_loss=(.5*(updated_y-target_tensor).square().sum()).item()))
    changed_image = image.clone()
    changed_image[0,0,1,1] = 2
    diagonal_kernel = torch.tensor([[1.,0.],[0.,-1.]],dtype=dtype)[None,None]
    contrast_fixtures = dict(
        edited_center_output=F.conv2d(changed_image,kernel)[0,0].tolist(),
        diagonal_output=F.conv2d(image,diagonal_kernel)[0,0].tolist(),
        diagonal_edited_output=F.conv2d(changed_image,diagonal_kernel)[0,0].tolist(),
        channel_task_inputs=[[2,3],[-1,4],[0,0]],
        channel_task_outputs=[[5,-1],[3,-5],[0,0]],
        practice_receptive_offsets=sorted(set(a+b for a in (-1,0,1) for b in (-4,0,4))))
    return dict(patch=dict(input=image[0,0].tolist(),kernel=kernel[0,0].tolist(),
                           output=F.conv2d(image,kernel)[0,0].tolist(),
                           unfolded=patches[0].tolist(),folded=folded[0,0].tolist(),
                           overlap_count=counts[0,0].tolist()),
                parity=parity,shared_update=shared_update,adjoint=adjoint,pooling=pooling,
                translation=translation,receptive_trace=trace,linear_profiles=profiles,
                dilation_offsets=dict(two_dilation_two=hole_offsets,one_then_two=filled_offsets),
                transposed_coverage=transposed,batchnorm_fold_max_error=fold_error,
                temperature_laplacian=float((laplace*temperature).sum()),
                shape_cases=shape_cases,nulls=nulls,changed_updates=changed_updates,
                contrasts=contrast_fixtures)


class DigitCNN(nn.Module):
    def __init__(self,pooling="max",global_average=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding=1)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)
        self.pooling = pooling
        self.global_average = global_average
        self.head = nn.Linear(16 if global_average else 64,10)

    def features(self,x):
        pool = F.max_pool2d if self.pooling == "max" else F.avg_pool2d
        first = F.relu(self.conv1(x))
        second = F.relu(self.conv2(pool(first,2)))
        final = pool(second,2)
        return first,second,final

    def forward(self,x):
        final = self.features(x)[-1]
        features = final.mean((2,3)) if self.global_average else final.flatten(1)
        return self.head(features)


def load_data():
    data = np.genfromtxt(HERE/"digits-400.csv",delimiter=",",names=True)
    images = np.column_stack([data[f"pixel_{i}"] for i in range(64)]).astype(np.float32)/16
    targets = data["digit"].astype(np.int64)
    train,val = train_test_split(np.arange(len(data)),test_size=.3,stratify=targets,random_state=22)
    return torch.from_numpy(images.reshape(-1,1,8,8)),torch.from_numpy(targets),train,val,data["source_id"].astype(int)


def metrics(model,x,y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return dict(cross_entropy=F.cross_entropy(logits,y).item(),
                    correct=(logits.argmax(1)==y).sum().item(),count=len(y))


def experiments():
    images,targets,train,val,source_ids = load_data()
    records = []
    configs = ("mlp","cnn_max","cnn_average","cnn_global_average")
    for seed in (1,2,3):
        for name in configs:
            torch.manual_seed(seed)
            if name == "mlp":
                model = nn.Sequential(nn.Flatten(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,10))
            else:
                model = DigitCNN("average" if name == "cnn_average" else "max",
                                 name == "cnn_global_average")
            optimizer = torch.optim.Adam(model.parameters(),lr=.003)
            trace = []
            for step in range(401):
                if step in (0,1,25,100,200,400):
                    trace.append(dict(step=step,train=metrics(model,images[train],targets[train]),
                                      validation=metrics(model,images[val],targets[val])))
                if step == 400:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images[train]),targets[train])
                loss.backward()
                optimizer.step()
            shifted_right = torch.zeros_like(images[val])
            shifted_right[:,:,:,1:] = images[val][:,:,:,:-1]
            shifted_down = torch.zeros_like(images[val])
            shifted_down[:,:,1:,:] = images[val][:,:,:-1,:]
            record = dict(name=name,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                          trace=trace,shifted_right=metrics(model,shifted_right,targets[val]),
                          shifted_down=metrics(model,shifted_down,targets[val]))
            if seed == 1 and name.startswith("cnn"):
                model.eval()
                visual_rows = []
                for index in val[:3]:
                    probe = images[index:index+1].clone().requires_grad_(True)
                    logits = model(probe)
                    predicted = logits.argmax(1).item()
                    gradient = torch.autograd.grad(logits[0,predicted],probe)[0]
                    first,second,final = model.features(probe.detach())
                    visual_rows.append(dict(source_id=int(source_ids[index]),actual=targets[index].item(),
                                            predicted=predicted,input=probe[0,0].detach().tolist(),
                                            probabilities=logits.softmax(1)[0].detach().tolist(),
                                            first_maps=first[0].detach().tolist(),
                                            second_maps=second[0].detach().tolist(),
                                            final_maps=final[0].detach().tolist(),
                                            logit_input_gradient=gradient[0,0].tolist()))
                record["visual_rows"] = visual_rows
                record["first_layer_kernels"] = model.conv1.weight[:,0].detach().tolist()
            records.append(record)
            print(name,seed,record["parameters"],trace[-1]["validation"],flush=True)
    return dict(split_seed=22,train_source_ids=source_ids[train].tolist(),
                validation_source_ids=source_ids[val].tolist(),fits=records)


if __name__ == "__main__":
    output = dict(versions=dict(python=platform.python_version(),numpy=np.__version__,
                                 sklearn=sklearn.__version__,torch=torch.__version__),
                  fixtures=fixtures(),experiment=experiments())
    (HERE/"calculated-inputs.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
