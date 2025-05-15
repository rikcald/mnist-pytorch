import torch
from torch import optim
from model import Mnist_Logistic
from load_data import load_mnist_data
from tqdm import tqdm
import torch.nn.functional as F

# Manually setting a seed will ensure that random values will be the same
torch.manual_seed(42)

# Setting our model as Mnist logistic - > softmax regression
model = Mnist_Logistic()
optimizer = optim.SGD(model.parameters(), lr=0.05)
loss_function = F.cross_entropy


train_loader, valid_loader, test_loader = load_mnist_data()
print(f"MNIST loaded with: {len(train_loader)} batches")

epochs = 5
batch_size = train_loader.batch_size

for epoch in range(epochs):
    # model.train()
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        pred = model(xb.view(-1, 784))
        loss = loss_function(pred, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # model.eval()
    with torch.no_grad():
        valid_loss = sum(
            loss_function(model(xb.view(-1, 784)), yb) for xb, yb in valid_loader
        ) / len(valid_loader)
    print(
        f"Epoch {epoch + 1}, Loss: {loss.item():.4f} , Validation Loss: {valid_loss:.4f}"
    )
