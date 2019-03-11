import torch
from torch import nn
from torch.autograd import Variable
from itertools import islice
from .model import LinearAutoencoder, ReluAutoencoder


class Trainer:
    def __init__(self, num_epochs, num_samples, dataloader, learning_rate, model_name):
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.optimizer = None
        self.criterion = None
        self.model = None
        self.model_name = model_name
        self.prepare(self.model_name)

    def prepare(self, model_name):
        if model_name == 'linear_autoencoder':
            self.model = LinearAutoencoder()
        elif model_name == 'relu_autoencoder':
            self.model = ReluAutoencoder()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def train(self):
        for epoch in range(self.num_epochs):
            for data in islice(self.dataloader, self.num_samples):
                img, _ = data
                img = img.view(img.size(0), -1)
                img = Variable(img)
                # ===================forward=====================
                output = self.model(img)
                loss = self.criterion(output, img)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.num_epochs, loss.data.item()))

        model_filename = './' + self.model_name + '.pth'
        torch.save(self.model.state_dict(), model_filename)

    def get_model(self):
        return self.model