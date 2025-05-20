import torch
from torch import nn
from src.load_data import load_mnist_data
from src.model import Mnist_Logistic, Mnist_CNN, Lambda, preprocess
from src.train import fit, evaluate
import torch.nn.functional as F
from torch import optim


def main():
    # data loading
    torch.manual_seed(42)
    train_loader, valid_loader, test_loader = load_mnist_data()
    print(f"MNIST loaded with: {len(train_loader)} batches")

    # choosing model and optimizer

    # model = Mnist_Logistic()
    # optimizer = optim.SGD(model.parameters(), lr=0.05)
    # model = Mnist_CNN()

    model = nn.Sequential(
        Lambda(preprocess),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # training and testing
    test_loss_beforeTraining = evaluate(model, F.cross_entropy, test_loader)
    print(f"Test_loss : {test_loss_beforeTraining:.4f}")
    fit(5, model, F.cross_entropy, optimizer, train_loader, valid_loader)
    test_loss = evaluate(model, F.cross_entropy, test_loader)
    print(f"Test_loss : {test_loss:.4f}")


if __name__ == "__main__":
    main()
