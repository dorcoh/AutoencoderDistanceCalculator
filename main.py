from aadc.data import Data
from aadc.trainer import Trainer
from aadc.distance_calculator import DistanceCalculator
from aadc.metrics import compute_score
import sys


def main(argv):
    try:
        if argv[0] == 'a':
            model_name = 'linear_autoencoder'
        elif argv[0] == 'b':
            model_name = 'relu_autoencoder'
    except:
        print('usage: python main.py <model_key>')
        print('a - linear autoencoder')
        print('b - autoencoder with relu activation functions')
        sys.exit(1)

    # dev
    num_samples = 15
    num_epochs = 1
    batch_size = 1
    learning_rate = 1e-3
    top_n_elemnts = [5, 10, 15, 20]

    data = Data(batch_size=batch_size)
    dataloader = data.get_datalodaer()

    trainer = Trainer(num_epochs=num_epochs, num_samples=num_samples, learning_rate=learning_rate,
                      dataloader=dataloader, model_name=model_name)

    trainer.train()
    model = trainer.get_model()

    calc = DistanceCalculator(num_samples=num_samples)
    calc.evaluate_model(dataloader, model.encoder)
    calc.compute_distances()
    origin, encoded = calc.get_distances()
    for n in top_n_elemnts:
        compute_score(origin, encoded, n)


if __name__ == '__main__':
    main(sys.argv[1:])