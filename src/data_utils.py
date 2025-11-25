import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, batch_size):
    """
    Returns train_loader, test_loader, num_classes, input_channels
    Forces all inputs to 3 channels (RGB) and 32x32 resolution.
    """
    dataset_name = dataset_name.lower()
    
    # Transform for Grayscale -> Fake RGB
    transform_gray_to_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # DUPLICATES CHANNELS
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3-channel norm
    ])

    if dataset_name == "mnist":
        train_d = datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray_to_rgb)
        test_d = datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray_to_rgb)
        num_classes = 10

    elif dataset_name == "fmnist" or dataset_name == "fashionmnist":
        train_d = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray_to_rgb)
        test_d = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray_to_rgb)
        num_classes = 10
        
    elif dataset_name == "cifar10":
        # CIFAR is already 3 channels, 32x32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_d = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_d = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=batch_size, shuffle=False)
    
    # We now always return 3 channels
    return train_loader, test_loader, num_classes, 3