import torch
from tqdm import tqdm
import numpy as np


def step(model, loss_func, xb, yb, opt=None):
    pred = model(xb)
    loss = loss_func(pred, yb)

    # this condition skips the backprop if there is no optmizer (e.g. for a validation set loss function)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_function, optimizer, train_loader, valid_loader):
    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            xb_flat = xb.view(-1, 784)  # from [bs,1,28,28] to [bs,784]
            loss, _ = step(model, loss_function, xb_flat, yb, optimizer)

        model.eval()
        valid_loss = evaluate(model, loss_function, valid_loader)
        print(
            f"Epoch {epoch + 1}, Loss: {loss:.4f} , Validation Loss: {valid_loss:.4f}"
        )


def evaluate(model, loss_function, data_loader):
    model.eval()
    with torch.no_grad():
        # - step() returns (loss.item(), len(xb)) for each batch.
        # - the input to zip() is [(loss1, size1), (loss2, size2), ...].
        # - the output is two tuples: (loss1, loss2, ...) and (size1, size2, ...).
        losses, batch_sizes = zip(
            *[
                step(model, loss_function, xb.view(-1, 784), yb)
                for xb, yb in data_loader
            ]
        )
        # Using NumPy for weighted average since losses and batch sizes are Python floats and ints, not PyTorch tensors
        return np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes)
