import torch
import plotly.express as px
import numpy as np
from load_data import load_mnist_data
import torchvision

# Run this code to plot a random digit from the dataset
train_dl, _, _ = load_mnist_data()


batch = next(iter(train_dl))
# images.size - > [bs , 1 , 28 , 28]
images, labels = batch


def showimg(imgBatch):
    # the slicing with ... is the same as [:rowsécols, : , : , :]
    rows = 1
    cols = imgBatch.shape[0]  # entire bs
    imgBatch = images[: rows * cols, :, :, :]
    resolved_grid = torchvision.utils.make_grid(
        imgBatch, padding=4, nrow=cols, normalize=True, value_range=(0, 1)
    )
    # permute [C,H,W] -> [H,W,C] since px.imshow works with this disposition
    fig = px.imshow(resolved_grid.permute(1, 2, 0))
    fig.update_layout(title="epoch -1")
    fig.show()


showimg(images)
