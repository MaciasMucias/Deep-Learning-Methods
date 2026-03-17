import torch
import time
from project1_cinic10.models import MODEL_REGISTRY

device = torch.device("cuda")
model = MODEL_REGISTRY["mobilenetv2"](dropout=0).to(device)
torch.backends.cudnn.benchmark = True
model.train()

optimizer = torch.optim.AdamW(model.parameters())
x = torch.randn(64, 3, 32, 32, device=device)
y = torch.randint(0, 10, (64,), device=device)
criterion = torch.nn.CrossEntropyLoss()

# warmup
for _ in range(5):
    optimizer.zero_grad()
    criterion(model(x), y).backward()
    optimizer.step()

# forward only
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()
print(f"forward only: {time.time()-start:.2f}s")

# forward + backward
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    optimizer.zero_grad()
    criterion(model(x), y).backward()
    optimizer.step()
torch.cuda.synchronize()
print(f"forward+backward: {time.time()-start:.2f}s")