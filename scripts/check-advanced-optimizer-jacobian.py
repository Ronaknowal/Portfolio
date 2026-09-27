"""Independent second derivative behind the nonlinear curvature illustration."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
theta = torch.tensor(1., dtype=torch.float64, requires_grad=True)
loss = torch.nn.functional.softplus(theta * theta) - theta * theta
gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
hessian = torch.autograd.grad(gradient, theta)[0].item()
probability = torch.sigmoid(theta * theta).item()
ggn = 4 * probability * (1 - probability)
extra = 2 * (probability - 1)
assert abs(hessian - (ggn + extra)) < 1e-14
result = {'passed': True, 'p': probability, 'GGN': ggn, 'extra': extra,
          'fullHessian': hessian, 'checks': ['Independent torch second derivative for nonlinear Jacobian illustration; GGN plus residual second derivative equals full Hessian']}
destination = Path('docs/teaching/deep-learning-completion/advanced-optimizers-lion-sophia-prodigy-schedule-free/jacobian-check.json')
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
