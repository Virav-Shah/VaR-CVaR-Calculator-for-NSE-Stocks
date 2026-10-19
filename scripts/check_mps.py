"""Verify actual accelerator forward/backward execution; fail if unavailable."""
import json
import platform
import torch

status = {'architecture': platform.machine(), 'torch': torch.__version__,
          'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()}
print(json.dumps(status, indent=2))
if not status['mps_available']:
    raise SystemExit('MPS unavailable in this process. CPU quantile models remain supported.')
x = torch.randn(128, 18, device='mps')
model = torch.nn.Linear(18, 6).to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
y = torch.randn(128, 1, device='mps')
q = torch.tensor([.95,.96,.97,.98,.99,.995], device='mps')
optimizer.zero_grad()
u = y - model(x)
loss = torch.maximum(q*u, (q-1)*u).mean()
loss.backward()
optimizer.step()
torch.mps.synchronize()
print(f'MPS quantile-loss forward/backward/optimizer smoke test passed: {loss.item():.6f}')
