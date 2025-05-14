import torch
import math
from torch import nn


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(784, 10) / math.sqrt(784)
        )  # Xavier init.
        self.bias = nn.Parameter(torch.zeros(10))  # Initialize the bias with zeros

    def forward(self, xb):
        return xb @ self.weights + self.bias


"""
faster and refactorized version

class Mnist_Logistic(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(784, 10)

  def forward(self, xb):
    return self.lin(xb)
"""
