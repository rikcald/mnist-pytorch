from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_mnist_data(batch_size=64, valid_split=0.2):
    transform = transforms.Compose(
        [
            # transforming the data (images) to pytorch tensors
            transforms.ToTensor(),
            # normalizing the tensors, i.e. the distribution of values on each sample should have mean=0.1307 and stddev=0.3081, check notebook 4
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # defining train validation and test datasets
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # splitting full_train into train and validation
    train_size = int((1 - valid_split) * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset, [train_size, valid_size]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_dataloader doesn't need shuffling and a small batch size
    valid_dataloader = DataLoader(
        train_dataset, batch_size=batch_size * 2, shuffle=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader
