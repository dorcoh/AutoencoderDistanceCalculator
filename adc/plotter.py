import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from .data import get_datalodaer
import torch
from torch.autograd import Variable
from copy import copy
import numpy as np
import pandas as pd


def plot_encoded_results(encoder, model_name, num_samples, batch_size):
    results, labels_list = test_model(encoder, num_samples, batch_size)
    data = prepare_plotting(results, labels_list)
    df = pd.DataFrame(data)
    df.columns = ['x', 'y', 'label']
    groups = df.groupby('label')
    df['label'] = df['label'].astype('int')
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    ax.legend()

    plt.savefig(model_name)


def test_model(encoder, num_samples, batch_size):
    dataloader = get_datalodaer(batch_size=batch_size, num_samples=num_samples, normalize=True)
    labels_list = []
    i = 0
    for data in dataloader:
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

    return results, labels_list


def prepare_plotting(results, labels_list):
    # convert to numpy array [x, y, label]
    labels_list = [a.detach().cpu().numpy() for a in labels_list]
    labels = np.array(labels_list).reshape(-1,1)
    data = np.concatenate([results, labels], axis=1)

    return data