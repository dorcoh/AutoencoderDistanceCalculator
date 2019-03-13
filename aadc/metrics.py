import numpy as np


def compute_score(min_distances_a, min_distances_b, img_size):
    scores = []
    for key, value in min_distances_a.items():
        scores.append(len(set(value[:img_size]).intersection(set(min_distances_b[key][:img_size]))) / img_size)

    print('Score for size of ', img_size, ' ', np.average(scores))


def compute_estimators(origin, encoded):
    """gets origin and encoded distances and returns meand and std"""
    estimators = dict()
    for name, samples in {'origin': origin, 'encoded': encoded}.items():
        for dist_type, dist_vector in samples.items():
            estimators[name + '_' + dist_type + '_avg'] = np.average(dist_vector)
            estimators[name + '_' + dist_type + '_std'] = np.std(dist_vector)

    return estimators
