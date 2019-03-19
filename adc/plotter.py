import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from .data import get_datalodaer
import torch
from torch.autograd import Variable
from copy import copy
import numpy as np
from itertools import islice


def plot_encoded_results(encoder, model_name, num_samples, batch_size):
    results, labels = test_model(encoder, num_samples, batch_size)
    data = np.concatenate([results, labels.reshape(-1, 1)], axis=1)
    cmap = plt.get_cmap('Reds')
    fig, ax = plt.subplots(figsize=(20,10))
    for row in data:
        x = row[0]
        y = row[1]
        label = int(row[2])
        ax.scatter(x, y, label=label, cmap=cmap(label))

    plt.legend()
    plt.savefig(model_name)


def test_model(encoder, num_samples, batch_size):
    dataloader = get_datalodaer(batch_size=batch_size, normalize=True, shuffle=False)
    labels_list = []
    i = 0
    for data in islice(dataloader, num_samples):
        img, labels = data
        labels_list.append(labels)
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        # ===================forward=====================
        output = encoder(img)
        arr = output.detach().cpu().numpy()
        if i == 0:
            results = copy(arr)
        else:
            results = np.concatenate([results, arr])
        i += 1

    labels_list = [a.detach().cpu().numpy() for a in labels_list]
    return results, np.array(labels_list).squeeze()