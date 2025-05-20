import torch
import plotly.express as px
import numpy as np
from load_data import load_mnist_data

# Run this code to plot a random digit from the dataset
train_dl, _, _ = load_mnist_data()


batch = next(iter(train_dl))
images, labels = batch


mnist_example = images[0][0].numpy()

print("A MNIST sample has size", mnist_example.shape)

fig = px.imshow(mnist_example)
fig.update_layout(title=f"Etichetta: {labels[0].item()}")
fig.show()
