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
    return loss.item(), len(xb), pred


def fit(epochs, model, loss_function, optimizer, train_loader, valid_loader):
    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            xb_flat = xb.view(-1, 784)  # from [bs,1,28,28] to [bs,784]
            loss, _, _ = step(model, loss_function, xb_flat, yb, opt=optimizer)

        model.eval()
        valid_loss, accuracy = evaluate(model, loss_function, valid_loader)
        print(
            f"Epoch {epoch + 1}, Loss: {loss:.4f} , Validation Loss: {valid_loss:.4f} , Accuracy : {accuracy * 100:.2f}%"
        )


# evaluate will return weighted average of the loss and the accuracy of an entire epoch
def evaluate(model, loss_function, data_loader):
    model.eval()
    with torch.no_grad():
        losses = []
        batch_sizes = []
        total_accuracy = []
        for xb, yb in data_loader:
            # evaluating weighted average loss
            loss, batch_size, pred = step(model, loss_function, xb.view(-1, 784), yb)
            losses.append(loss)
            batch_sizes.append(batch_size)
            total_accuracy.append(evaluate_accuracy(pred, yb))
        #  Using NumPy for weighted average since losses and batch sizes are Python floats and ints, not PyTorch tensors
        return np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes), np.sum(
            total_accuracy
        ) / len(total_accuracy)
        print(
            f"Test_loss : {np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes):.4f} | Accuracy : {(np.sum(total_accuracy) / len(total_accuracy)) * 100:.2f}%"
        )


def evaluate_accuracy(pred, yb):
    predicted_classes_perBatch = pred.argmax(dim=1)
    correct_predictions_perBatch = (predicted_classes_perBatch == yb).sum().item()
    # incorrect_predictions = len(yb) - correct_predictions_perBatch
    accuracy_perBatch = correct_predictions_perBatch / len(yb)
    print(
        f"correct predictions : {correct_predictions_perBatch} / {len(yb)} | accuracy per batch : {accuracy_perBatch * 100:.2f}%"
    )
    return accuracy_perBatch


# TODO riportare la media di correct predictions e accuracy
# TODO per ogni epoch mostrare qualche random image missclassificata
