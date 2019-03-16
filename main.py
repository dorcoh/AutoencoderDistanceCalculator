from adc.trainer import Trainer
from adc.distance_calculator import DistanceCalculator
from adc.metrics import compute_score, compute_estimators
from adc.loader import test_load_distances
from adc.utils import pprint_dict
from adc.plotter import plot_encoded_results
import sys


def main(argv):
    try:
        model_key = argv[0]
        loss_function = argv[1]

        if loss_function not in ['L1', 'MSE']:
            raise Exception

        if model_key == 'a':
            model_name = 'linear_autoencoder' + '_' + loss_function
        elif model_key == 'b':
            model_name = 'relu_autoencoder' + '_' + loss_function

    except:
        print('usage: python main.py <model_key> <loss_function>')
        print('model keys: ')
        print('a - linear autoencoder')
        print('b - autoencoder with relu activation functions')
        print('loss functions: L1 or MSE')
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

    calc = DistanceCalculator(num_samples=num_samples, batch_size=batch_size, model_name=model_name)
    calc.evaluate(model.encoder)
    calc.compute_distances()
    calc.save_distances()

    origin, encoded = calc.get_distances()
    estimators = compute_estimators(origin, encoded)
    pprint_dict(estimators)


if __name__ == '__main__':
    main(sys.argv[1:])
