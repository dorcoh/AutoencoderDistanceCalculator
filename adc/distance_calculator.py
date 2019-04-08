import pickle

from torch.autograd import Variable
from copy import copy
import numpy as np
from scipy.spatial import distance
from .data import get_datalodaer
import torch


class DistanceCalculator:
    def __init__(self, num_samples, batch_size, model_name):
        self.num_samples = num_samples
        self.model_name = model_name

        self.results = None
        self.origin = None

        self.dataloader = get_datalodaer(batch_size, num_samples)
        self.dataloader_normalized = get_datalodaer(batch_size, num_samples, normalize=True)

        self.original_distances = {}
        self.encoded_distances = {}

    def _get_original_samples(self):
        i = 0
        labels = []
        for data in self.dataloader:
            img, labels_tensor = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            flat_img = img.detach().cpu().numpy()
            flat_labels = labels_tensor.detach().cpu().numpy()
            if i == 0:
                origin = copy(flat_img)
                labels = copy(flat_labels)
            else:
                origin = np.concatenate([origin, flat_img])
                labels = np.concatenate([labels, flat_labels])
            i += 1

        self.origin = origin
        self.labels = dict(enumerate(labels.flatten()))

    def _evaluate_model(self, encoder):
        i = 0
        for data in self.dataloader_normalized:
            img, labels = data
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

        self.results = results

    def evaluate(self, encoder):
        self._get_original_samples()
        self._evaluate_model(encoder)

    def _compute_distance(self, elements, size=200):
        distances_dict = dict()
        distances_dict['cityblock'] = self._compute_norm(elements, norm=1, size=size)
        distances_dict['euclidean'] = self._compute_norm(elements, norm=2, size=size)
        return distances_dict

    def _compute_norm(self, elements, size=200, norm=2):
        """
        elements: concatenated array of N samples with shape (N, dimension)
        """
        min_distances_idx = {}
        min_distances_values = {}
        min_distances_labels = {}

        for idx_f, item_f in enumerate(elements):
            distances_tmp = []
            for item_s in elements:
                distances_tmp.append(float(np.linalg.norm(item_f - item_s, ord=norm)))

            min_distances_idx[idx_f] = sorted(range(len(distances_tmp)), key=lambda i: distances_tmp[i])[1:size]
            min_distances_values[idx_f] = sorted(distances_tmp)[1:size]
            min_distances_labels[idx_f] = [self.labels[i] for i in min_distances_idx[idx_f]]

        return min_distances_idx, min_distances_values, min_distances_labels

    def _compute_origin_distance(self):
        self.original_distances = self._compute_distance(self.origin)

    def _compute_encoded_distance(self):
        self.encoded_distances = self._compute_distance(self.results)

    def compute_distances(self):
        self._compute_origin_distance()
        self._compute_encoded_distance()

    def get_distances(self):
        return self.original_distances, self.encoded_distances

    @staticmethod
    def _save_distances(distances, name):
        with open('{pickle_name}.pkl'.format(pickle_name=name), 'wb') as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_distances(self):
        original_name = self.model_name + '_original_distances'
        encoded_name = self.model_name + '_encoded_distances'
        self._save_distances(self.original_distances, name=original_name)
        self._save_distances(self.encoded_distances, name=encoded_name)
