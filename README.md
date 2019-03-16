# AutoencoderDistanceCalculator WIP

### Requirements

- pytorch
- numpy
- matplotlib

### Usage

`python main.py <model_key> <original_distances_fname> <encoded_distances_fname> <loss_function>`
- a - linear autoencoder
- b - autoencoder with relu activation functions
- Supported loss functions: L1 or MSE