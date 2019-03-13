from aadc.trainer import Trainer
from aadc.distance_calculator import DistanceCalculator
from aadc.metrics import compute_score, compute_estimators
from aadc.loader import test_load_distances
from aadc.utils import pprint_dict
import sys


def main(argv):
    try:
        if argv[0] == 'a':
            model_name = 'linear_autoencoder'
        elif argv[0] == 'b':
            model_name = 'relu_autoencoder'
        origin_fname = argv[1]
        encoded_fname = argv[2]
        print('Original Distances will be saved to: ', argv[1])
        print('Encoded Distances will be saved to: ', argv[2])
    except:
        print('usage: python main.py <model_key> <original_distances_filename> <encoded_distances_filename>')
        print('model keys: ')
        print('a - linear autoencoder')
        print('b - autoencoder with relu activation functions')
        sys.exit(1)

    # dev
    num_samples = 5
    num_epochs = 1
    batch_size = 1
    learning_rate = 1e-3
    top_n_elemnts = [5, 10, 15, 20]

    trainer = Trainer(num_epochs=num_epochs, num_samples=num_samples, batch_size=batch_size,
                      learning_rate=learning_rate, model_name=model_name)

    trainer.train()
    model = trainer.get_model()

    calc = DistanceCalculator(num_samples=num_samples, batch_size=batch_size)
    calc.evaluate(model.encoder)
    calc.compute_distances()
    calc.save_distances(original_name=origin_fname, encoded_name=encoded_fname)

    test_load_distances(original_fname=origin_fname+'.pkl', encoded_fname=encoded_fname+'.pkl')

    origin, encoded = calc.get_distances()
    estimators = compute_estimators(origin, encoded)
    pprint_dict(estimators)
    compute_score(origin, encoded, 10)
    # for n in top_n_elemnts:
    #     compute_score(origin, encoded, n)

    # TODO: scores, plots (after encoding)


if __name__ == '__main__':
    main(sys.argv[1:])
