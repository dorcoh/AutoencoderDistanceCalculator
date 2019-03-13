import pickle

from torch.autograd import Variable
from copy import copy
import numpy as np
from itertools import islice
from scipy.spatial import distance
from .data import get_datalodaer


class DistanceCalculator:
    def __init__(self, num_samples, batch_size):
        self.num_samples = num_samples

        self.results = None
        self.origin = None

        self.dataloader = get_datalodaer(batch_size)
        self.dataloader_normalized = get_datalodaer(batch_size, normalize=True)

        self.original_distances = {}
        self.encoded_distances = {}

    def _get_original_samples(self):
        i = 0
        for data in islice(self.dataloader, self.num_samples):
            img, labels = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            flat_img = img.detach().numpy()
            if i == 0:
                origin = copy(flat_img)
            else:
                origin = np.concatenate([origin, flat_img])
            i += 1

        self.origin = origin

    def _evaluate_model(self, encoder):
        i = 0
        for data in islice(self.dataloader_normalized, self.num_samples):
            img, labels = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            # ===================forward=====================
            output = encoder(img)
            arr = output.detach().numpy()
            if i == 0:
                results = copy(arr)
            else:
                results = np.concatenate([results, arr])
            i += 1

        self.results = results

    def evaluate(self, encoder):
        self._get_original_samples()
        self._evaluate_model(encoder)

    @staticmethod
    def _compute_distance(elements):
        distances_dict = dict()
        distances_dict['cityblock'] = distance.pdist(elements, 'cityblock')
        distances_dict['cityblock_avg'] = np.average(distances_dict['cityblock'])
        distances_dict['cityblock_std'] = np.std(distances_dict['cityblock'])
        distances_dict['euclidean'] = distance.pdist(elements, 'euclidean')
        distances_dict['euclidean_avg'] = np.average(distances_dict['euclidean'])
        distances_dict['euclidean_std'] = np.std(distances_dict['euclidean'])
        return distances_dict

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

    def save_distances(self, original_name, encoded_name):
        self._save_distances(self.original_distances, name=original_name)
        self._save_distances(self.encoded_distances, name=encoded_name)
