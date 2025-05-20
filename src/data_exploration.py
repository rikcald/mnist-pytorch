import torch
import plotly.express as px
import numpy as np 
from load_data import load_mnist_data

train_dl,_,_ = load_mnist_data()
print(f"train_dl is a {type(train_dl)}")

batch = next(iter(train_dl))
images, labels = batch


print(f"Shape delle immagini: {images.shape}")   # es: torch.Size([64, 1, 28, 28])
print(f"Shape delle etichette: {labels.shape}")  # es: torch.Size([64])
print(f"Tipo delle immagini: {type(images)}")    # torch.Tensor
print(f"Etichette: {labels[:10]}")   

mnist_example = train_dl[42][0][0].numpy()
print('A MNIST sample has size', mnist_example.shape)
fig = px.imshow(mnist_example)
fig.show()