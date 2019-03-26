from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataset import Subset


def get_datalodaer(batch_size, num_samples, normalize=False, shuffle=False):
    """returns pytorch mnist dataloader"""
    transform = [transforms.ToTensor()]
    if normalize:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img_transform = transforms.Compose(transform)

    indices = [i for i in range(num_samples)]
    dataset = Subset(MNIST('./data', transform=img_transform, download=True), indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader