import numpy as np


def compute_score(min_distances_a, min_distances_b, img_size):
    scores = []
    for key, value in min_distances_a.items():
        scores.append(len(set(value[:img_size]).intersection(set(min_distances_b[key][:img_size]))) / img_size)

    print('Score for size of ', img_size, ' ', np.average(scores))