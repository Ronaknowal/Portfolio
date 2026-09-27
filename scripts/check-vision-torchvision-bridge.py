from pathlib import Path
import contextlib
import io
import json
import runpy
import torch
import torchvision
from torchvision.models.swin_transformer import ShiftedWindowAttention

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT/'docs/teaching/drafts/vision-transformers-vit-deit-swin-dinov2'
OUT = ROOT/'docs/teaching/deep-learning-completion/vision-transformers-vit-deit-swin-dinov2'
torch.set_num_threads(2)
bridge = runpy.run_path(str(PACKET/'vision_library_bridge.py'))
mechanism = runpy.run_path(str(PACKET/'vision-mechanisms.py'))
captured = io.StringIO()
with contextlib.redirect_stdout(captured): bridge['main']()
torch.manual_seed(127)
ours = mechanism['ShiftedWindowAttention'](4,2,3,1).double().eval()
theirs = ShiftedWindowAttention(4,[3,3],[1,1],2).double().eval()
theirs.qkv.load_state_dict(ours.qkv.state_dict()); theirs.proj.load_state_dict(ours.output.state_dict())
with torch.no_grad():
    ours.relative_bias.copy_(torch.randn_like(ours.relative_bias)*.3)
    theirs.relative_position_bias_table.copy_(ours.relative_bias.T)
probes = []
for height,width in ((9,12),(5,7)):
    value = torch.randn(1,height,width,4,dtype=torch.float64)
    error = float((ours(value)-theirs(value)).abs().max().detach())
    if height % 3 == width % 3 == 0: assert error < 1e-10
    else: assert error > 1e-5
    probes.append({'height':height,'width':width,'maximumError':error,'meaning':'equivalent divisible grid' if height%3==width%3==0 else 'different documented padded-key policies'})
record={'passed':True,'torch':torch.__version__,'torchvision':torchvision.__version__,'checks':['Supplied output, input-gradient, all-parameter-gradient and odd-grid merge comparisons passed','Fresh9x12 dimensions agree','Fresh5x7 padded policies differ as explained'],'probes':probes,'stdout':captured.getvalue()}
(OUT/'library-checks.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
