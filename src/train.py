import torch
from torch import optim
from model import Mnist_Logistic
from load_data import load_mnist_data
from tqdm import tqdm

torch.manual_seed(42)
model = Mnist_Logistic()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = torch.nn.CrossEntropyLoss()

train_loader, test_loader = load_mnist_data()
print(f"MNIST loaded with: {len(train_loader)} batches")

epochs = 5
batch_size = train_loader.batch_size
"""
        - scorrere ogni batch
          pred(train) - > forward()
          loss = loss_func(pred,test_loader)
          backprop 
          with no grad
            - aggiorno i pesi
            -azzero i pesi
    """
for epoch in range(epochs):
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        pred = model(xb.view(-1, 784))
        loss = loss_function(pred, yb)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
