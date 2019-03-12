from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_datalodaer(batch_size, normalize=False, shuffle=False):
    """gets pytorch mnist dataloader"""
    transform = [transforms.ToTensor()]
    if normalize:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img_transform = transforms.Compose(transform)

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader