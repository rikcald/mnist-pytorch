import torch
from src.load_data import load_mnist_data
from src.model import Mnist_Logistic, Mnist_CNN
from src.train import fit
import torch.nn.functional as F
from torch import optim


def main():
    torch.manual_seed(42)
    train_loader, valid_loader, test_loader = load_mnist_data()
    print(f"MNIST loaded with: {len(train_loader)} batches")

    # model = Mnist_Logistic()
    # optimizer = optim.SGD(model.parameters(), lr=0.05)

    model = Mnist_CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    fit(10, model, F.cross_entropy, optimizer, train_loader, valid_loader)


if __name__ == "__main__":
    main()
