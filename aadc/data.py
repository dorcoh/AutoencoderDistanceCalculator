from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None

    def get_datalodaer(self):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = MNIST('./data', transform=img_transform, download=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return self.dataloader