import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from itertools import islice
from .model import LinearAutoencoder, ReluAutoencoder
from .data import get_datalodaer
from .utils import to_img
import os


class Trainer:
    def __init__(self, num_epochs, num_samples, batch_size, learning_rate, model_name, loss='MSE'):
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dataloader = get_datalodaer(batch_size, normalize=True, shuffle=True)
        self.learning_rate = learning_rate
        self.optimizer = None
        self.criterion = None
        self.model = None
        self.model_name = model_name
        self.loss = loss
        self.prepare(self.model_name)

    def prepare(self, model_name):
        if 'linear_autoencoder' in model_name:
            self.model = LinearAutoencoder()
        elif 'relu_autoencoder' in model_name:
            self.model = ReluAutoencoder()
        if self.loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.loss == 'L1':
            self.criterion = nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def train(self, save_imgs_flag=True):
        print("Started training for model name: ", self.model_name)
        for epoch in range(self.num_epochs):
            for data in islice(self.dataloader, self.num_samples):
                img, _ = data
                img = img.view(img.size(0), -1)
                if torch.cuda.is_available():
                    img = Variable(img).cuda()
                else:
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

            if save_imgs_flag and epoch % 10 == 0:
                self.save_images(output, epoch)

        model_filename = './' + self.model_name + '.pth'
        torch.save(self.model.state_dict(), model_filename)

    def get_model(self):
        return self.model

    def save_images(self, output, epoch, dir='decoded_images'):
        path = './' + dir
        if not os.path.exists(path):
            os.mkdir(path)
        pic = to_img(output.cpu().data)
        save_image(pic, path + '/{}_image_{}.png'.format(self.model_name, epoch))
