from src.load_data import load_mnist_data


def main():
    train_loader, test_loader = load_mnist_data()
    print(f"MNIST loaded with: {len(train_loader)} batch")


if __name__ == "__main__":
    main()
