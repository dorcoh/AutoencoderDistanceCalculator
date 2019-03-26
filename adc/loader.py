import pickle
from scipy.spatial import distance
import torch


def load_pickle(filename):
    with open(filename, 'rb') as handler:
        content = pickle.load(handler)

    return content


def load_distances(filename):
    """returns dictionary with key as distance type and value as square form distance matrix"""
    distances = load_pickle(filename)
    for key, dist in distances.items():
        if 'avg' in key or 'std' in key:
            continue
        distances[key] = distance.squareform(dist)

    return distances


def test_load_distances(original_fname='dev_orig.pkl', encoded_fname='dev_encd.pkl'):
    """pretty print computations"""
    print("original")
    for key, value in load_distances(original_fname).items():
        print(key)
        print(value)

    print()
    print("encoded")
    for key, value in load_distances(encoded_fname).items():
        print(key)
        print(value)


def save_checkpoint(model, optimizer, epoch, filepath):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filepath)


def load_checkpoint(model, optimizer, filepath):
    state = torch.load(filepath, lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch']

    return model, optimizer, epoch