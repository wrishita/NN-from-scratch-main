# Neural Network from Scratch

This project implements a simple neural network for digit recognition using only NumPy and Pandas. The implementation builds a feedforward neural network with a single hidden layer to classify handwritten digits from the MNIST dataset.

## Overview

This neural network is built entirely from first principles, without relying on any deep learning frameworks like TensorFlow or PyTorch. The implementation includes:

- Forward and backward propagation algorithms
- Leaky ReLU activation function for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Gradient descent optimizer

## Network Architecture

The network has the following architecture:
- Input layer: 784 neurons (28x28 pixel images flattened)
- Hidden layer: 10 neurons with Leaky ReLU activation
- Output layer: 10 neurons with Softmax activation

## Dependencies

- NumPy
- Pandas
- Matplotlib

## Dataset

The project uses the MNIST dataset for training and evaluation. The dataset consists of 28x28 grayscale images of handwritten digits (0-9).

## Implementation Details

### Data Preprocessing
- Loads and shuffles the dataset
- Splits the data into training and development sets
- Normalizes pixel values to range [0,1]

### Neural Network Components
- **Parameter Initialization**: Implements Xavier/He initialization for weights
- **Activation Functions**: Implements Leaky ReLU and Softmax
- **Forward Propagation**: Computes outputs through the network
- **Backward Propagation**: Computes gradients for parameter updates
- **Gradient Descent**: Updates parameters to minimize loss

### Training and Evaluation
- Implements gradient descent training loop
- Tracks loss and accuracy during training
- Provides visualization of training progress
- Evaluates performance on a development set

## Results

The model achieves good accuracy on the MNIST dataset without any complex architecture or regularization techniques, demonstrating the power of understanding the fundamentals of neural networks.

### Loss and Accuracy Curves

![Loss and Accuracy Curves](https://i.ibb.co/vxhXtbhM/image.png)

*The graph above shows the training loss decreasing and accuracy increasing over iterations, demonstrating successful learning by the neural network.*

## Usage

1. Ensure you have the MNIST dataset in a `./dataset/train.csv` format
2. Run the script to train the neural network
3. The model will display:
   - Training progress (accuracy every 10 iterations)
   - Loss and accuracy curves
   - Sample predictions with visualizations

## Sample Visualization

The code includes functionality to visualize:
- Loss and accuracy curves during training
- Sample digit predictions with their corresponding images

### Sample Predictions

![Sample Digit Prediction](https://i.ibb.co/5gMDCYgy/image.png)

*The image above shows an example of the model correctly predicting a handwritten digit from the test set.*

## Key Learnings

This project demonstrates:
- How neural networks work at their most fundamental level
- Implementation of forward and backward propagation algorithms
- The mathematics behind activation functions and loss calculations
- Gradient-based optimization techniques

## Future Improvements

Potential enhancements to the project:
- Add more hidden layers
- Implement dropout and other regularization techniques
- Try different activation functions
- Add batch training capabilities
- Implement different optimization algorithms (Adam, RMSprop, etc.)

## License

[MIT License](LICENSE)

## Acknowledgments

This project was created as a learning exercise to understand the inner workings of neural networks without relying on high-level frameworks.
