from adc.trainer import Trainer
from adc.distance_calculator import DistanceCalculator
from adc.metrics import compute_score, compute_estimators
from adc.loader import test_load_distances
from adc.utils import pprint_dict
from adc.plotter import plot_encoded_results
import sys


def main(argv):
    try:
        origin_fname = argv[1]
        encoded_fname = argv[2]
        loss_function = argv[3]

        if loss_function not in ['L1', 'MSE']:
            raise Exception

        if argv[0] == 'a':
            model_name = 'linear_autoencoder' + '_' + loss_function
        elif argv[0] == 'b':
            model_name = 'relu_autoencoder' + '_' + loss_function

        print('Original Distances will be saved to: ', argv[1])
        print('Encoded Distances will be saved to: ', argv[2])
    except:
        print('usage: python main.py <model_key> <original_distances_fname> <encoded_distances_fname> <loss_function>')
        print('model keys: ')
        print('a - linear autoencoder')
        print('b - autoencoder with relu activation functions')
        print('Valid loss functions are: L1 or MSE')
        sys.exit(1)

    num_samples = 60000
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    trainer = Trainer(num_epochs=num_epochs, num_samples=num_samples, batch_size=batch_size,
                      learning_rate=learning_rate, model_name=model_name, loss=loss_function)

    trainer.train()
    model = trainer.get_model()
    plot_encoded_results(model.encoder, model_name, num_samples, batch_size)

    calc = DistanceCalculator(num_samples=num_samples, batch_size=batch_size)
    calc.evaluate(model.encoder)
    calc.compute_distances()
    calc.save_distances(original_name=origin_fname, encoded_name=encoded_fname)

    # test_load_distances(original_fname=origin_fname+'.pkl', encoded_fname=encoded_fname+'.pkl')

    origin, encoded = calc.get_distances()
    estimators = compute_estimators(origin, encoded)
    pprint_dict(estimators)


if __name__ == '__main__':
    main(sys.argv[1:])
