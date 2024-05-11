# MNIST Neural Network from Scratch

This project implements a neural network from scratch to classify the MNIST handwritten digit dataset. The neural network is built using numpy for numerical computations and matplotlib.pyplot for data visualization. The model achieves an impressive accuracy of 99.7% on the test set.

![image](https://github.com/Kamran433/MNIST-Neural-Network/assets/102954239/52270a5d-6253-4aaa-a292-c69ad8455b96)

## Maths

1. **Input Layer (X):** The MNIST dataset consists of grayscale images of handwritten digits, each image being 28x28 pixels. This results in an input vector \(X\) of size 784 (28x28=784), where each element represents a pixel intensity (0-255).

2. **Weights (W) and Bias (b):** The neural network learns weights (\(W\)) and biases (\(b\)) for each neuron in the network. For the input layer to the hidden layer, \(W\) is a matrix of size [hidden_units, input_features] and \(b\) is a vector of size [hidden_units]. Similarly, for the hidden layer to the output layer, \(W\) is a matrix of size [output_units, hidden_units] and \(b\) is a vector of size [output_units].

3. **Activation Function (sigmoid, relu, softmax):** The hidden layers typically use the ReLU (Rectified Linear Unit) activation function, while the output layer uses the softmax activation function to produce probabilities for each class.

4. **Forward Propagation:**
   - \(Z^{[1]} = W^{[1]}X + b^{[1]}\) (for the hidden layer)
   - \(A^{[1]} = \text{ReLU}(Z^{[1]})\) (activation of the hidden layer)
   - \(Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}\) (for the output layer)
   - \(A^{[2]} = \text{Softmax}(Z^{[2]})\) (final output probabilities)

5. **Cost Function (Cross-Entropy):** The cross-entropy loss is commonly used for multi-class classification tasks like MNIST.
   \[ J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) \]
   where \(m\) is the number of examples, \(C\) is the number of classes, \(y_{i,c}\) is 1 if the example \(i\) belongs to class \(c\), and \(\hat{y}_{i,c}\) is the predicted probability that example \(i\) belongs to class \(c\).

6. **Backpropagation:**
   - Calculate the gradient of the cost function with respect to the weights and biases using the chain rule.
   - Update the weights and biases using gradient descent:
     \[ W^{[l]} = W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}} \]
     \[ b^{[l]} = b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}} \]
     where \(\alpha\) is the learning rate.

7. **Training:** Iterate through the dataset multiple times (epochs), updating the weights and biases after each batch of examples. The goal is to minimize the cost function \(J\) by adjusting the weights and biases.


## Features
- Custom implementation of a neural network using only numpy, and pandas library.
- Training and testing on the MNIST dataset
- Visualization of training progress and results using matplotlib.pyplot

## Requirements
- Python 3
- numpy
- pandas
- matplotlib

## Usage
1. Clone the repository: `git clone [https://github.com/Kamran433/MNIST-Neural-Network]`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the `main.py` script to train and test the neural network.

## Results
- Training accuracy: 99.8%
- Test accuracy: 99.7%

## Future Improvements
- Experiment with different network architectures and hyperparameters
- Implement more advanced optimization techniques such as Adam or RMSprop
- Explore other datasets and problems to apply the neural network to

Feel free to contribute, report issues, or suggest improvements!
