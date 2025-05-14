from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist_data(batch_size=64):
    transform = transforms.Compose(
        [
            # transforming the data (images) to pytorch tensors
            transforms.ToTensor(),
            # normalizing the tensors, i.e. the distribution of values on each sample should have mean=0.1307 and stddev=0.3081, check notebook 4
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # defining train and test dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
