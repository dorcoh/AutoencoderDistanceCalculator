from aadc.data import get_datalodaer
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

        print('Original Distances will be saved to: ', argv[1])
        print('Encoded Distances will be saved to: ', argv[2])
    except:
        print('usage: python main.py <model_key> <original_distances_filename> <encoded_distances_filename>')
        print('model keys: ')
        print('a - linear autoencoder')
        print('b - autoencoder with relu activation functions')
        sys.exit(1)

    # dev
    num_samples = 60000
    num_epochs = 50
    batch_size = 256
    learning_rate = 1e-3
    top_n_elemnts = [5, 10, 15, 20]

    trainer_dataloader = get_datalodaer(batch_size, normalize=True, shuffle=True)

    trainer = Trainer(num_epochs=num_epochs, num_samples=num_samples, learning_rate=learning_rate,
                      dataloader=trainer_dataloader, model_name=model_name)

    trainer.train()
    model = trainer.get_model()

    evaluator_dataloader = get_datalodaer(batch_size)
    calc = DistanceCalculator(num_samples=num_samples)
    calc.evaluate_model(evaluator_dataloader, model.encoder)

    calc.compute_distances()
    calc.save_distances(original_name=argv[1], encoded_name=argv[2])
    origin, encoded = calc.get_distances()
    for n in top_n_elemnts:
        compute_score(origin, encoded, n)


if __name__ == '__main__':
    main(sys.argv[1:])
