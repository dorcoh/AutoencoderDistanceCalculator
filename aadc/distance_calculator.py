import torch
from torch.autograd import Variable
from copy import copy
import numpy as np
from itertools import islice
from scipy.spatial import distance


class DistanceCalculator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.results = None
        self.origin = None

        self.original_distances = {}
        self.encoded_distances = {}

        self.original_min_distances = {}
        self.encoded_min_distances = {}

    def evaluate_model(self, dataloader, encoder):
        """evaluates and returns concatenated encoded images, and the originals"""
        i = 0
        for data in islice(dataloader, self.num_samples):
            img, labels = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            # ===================forward=====================
            output = encoder(img)
            arr = output.detach().numpy()
            flat_img = img.detach().numpy()
            if i == 0:
                results = copy(arr)
                origin = copy(flat_img)
            else:
                results = np.concatenate([results, arr])
                origin = np.concatenate([origin, flat_img])
            i += 1

        self.results = results
        self.origin = origin

    def compute_distance(self, elements, size=1000):
        """
        elements: concatenated array of N samples with shape (N, dimension)
        """
        distances = {}
        min_distances = {}
        for idx_f, item_f in enumerate(elements):
            distances[idx_f] = []
            for item_s in elements:
                distances[idx_f].append(float(np.linalg.norm(item_f - item_s)))

            if idx_f % 10000 == 0:
                print(idx_f)
            min_distances[idx_f] = sorted(range(len(distances[idx_f])), key=lambda i: distances[idx_f][i])[1:size]

        return min_distances, distances

    def compute_numpy_distances(self, elements):
        # TODO: check distances make sense, complete implementation
        # distances_matrix = distance.pdist(elements, lambda u, v: np.linalg.norm(u - v) )
        distances_matrix = distance.cdist(elements, elements, 'cityblock')
        sorted_indices = np.argsort(distances_matrix, axis=0)
        return distances_matrix[sorted_indices]

    def compute_origin_distance(self):
        self.original_distances, self.original_min_distances = self.compute_distance(self.origin)

    def compute_encoded_distance(self):
        self.encoded_distances, self.encoded_min_distances = self.compute_distance(self.results)

    def compute_distances(self):
        self.compute_origin_distance()
        self.compute_encoded_distance()

    def get_distances(self):
        return self.original_min_distances, self.encoded_min_distances
