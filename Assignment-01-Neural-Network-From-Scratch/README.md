# Neural Network Implementation from Scratch

## 1. Objective

Implement a simple **feedforward neural network from scratch in Python** without using built-in deep learning libraries.

The implementation focuses on the fundamental components of a neural network:

* Forward Pass
* Backward Propagation (Backpropagation)
* Loss Calculation
* Gradient Descent
* Model Training
* Prediction and Evaluation

The neural network is implemented using **Python and NumPy** and is trained to classify handwritten digits from the MNIST dataset.

---

## 2. Problem Definition

### Dataset

The **MNIST (Modified National Institute of Standards and Technology)** dataset is used for training and testing the neural network.

MNIST consists of grayscale images of handwritten digits from **0 to 9**. Each image has a size of **28 × 28 pixels**.

For this implementation:

* **Training samples used:** 10,000
* **Testing samples:** 10,000
* **Image size:** 28 × 28 pixels
* **Input features:** 784
* **Number of classes:** 10

Each 28 × 28 image is flattened into a 784-dimensional input vector before being passed to the neural network.

### Task

The task is a **multi-class classification problem**.

The neural network classifies each handwritten digit into one of the ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The integer labels are converted into **one-hot encoded vectors** for training.

---

## 3. Methodology

### 3.1 Neural Network Architecture

The implemented neural network consists of:

```text
Input Layer
784 neurons
     ↓
Hidden Layer
128 neurons
     ↓
Output Layer
10 neurons
```

### Network Configuration

| Component         | Configuration               |
| ----------------- | --------------------------- |
| Input neurons     | 784                         |
| Hidden neurons    | 128                         |
| Output neurons    | 10                          |
| Hidden activation | ReLU                        |
| Output activation | Softmax                     |
| Loss function     | Cross-Entropy               |
| Optimisation      | Mini-Batch Gradient Descent |
| Learning rate     | 0.01                        |
| Epochs            | 20                          |
| Batch size        | 64                          |

The input layer contains **784 neurons** because each 28 × 28 image is flattened into 784 input features.

The hidden layer contains **128 neurons** and uses the **ReLU activation function**.

The output layer contains **10 neurons**, corresponding to the ten MNIST digit classes, and uses **Softmax** to produce class probabilities.

### Weight Initialization

The weights are initialized using **He initialization**, which is suitable for networks using ReLU activation.

The biases are initialized to zero.

The random seed is set to `42` to make the initialization reproducible.

---

### 3.2 Forward Pass

The forward pass moves the input data through the neural network to produce the predicted class probabilities.

#### Hidden Layer

The weighted sum is first calculated:

```text
Z1 = XW1 + b1
```

The ReLU activation is then applied:

```text
A1 = ReLU(Z1)
```

where:

* `X` = input data
* `W1` = weights of the first layer
* `b1` = bias of the first layer
* `Z1` = weighted sum
* `A1` = activated hidden-layer output

#### Output Layer

The hidden-layer output is passed to the output layer:

```text
Z2 = A1W2 + b2
```

The Softmax function is then applied:

```text
A2 = Softmax(Z2)
```

The resulting `A2` contains the probability of the input image belonging to each of the ten digit classes.

The probabilities produced by Softmax sum to approximately **1**.

---

### 3.3 Backpropagation

Backpropagation is used to calculate the gradients of the loss function with respect to the network's weights and biases.

The error is propagated backwards from the output layer to the hidden layer.

For the output layer, the gradient is calculated as:

```text
dZ2 = A2 - y_true
```

The gradients for the weights and biases are then calculated:

```text
dW2 = A1ᵀ dZ2 / m
db2 = sum(dZ2) / m
```

The error is propagated to the hidden layer:

```text
dA1 = dZ2 W2ᵀ
```

The derivative of the ReLU activation function is then applied:

```text
dZ1 = dA1 × ReLU'(Z1)
```

Finally, the gradients for the first layer are calculated:

```text
dW1 = Xᵀ dZ1 / m
db1 = sum(dZ1) / m
```

These gradients are used to update the weights and biases during gradient descent.

---

### 3.4 Loss Function

The **Cross-Entropy Loss** function is used because this is a multi-class classification problem.

The cross-entropy loss measures the difference between the actual one-hot encoded labels and the predicted class probabilities.

The implementation also uses a small epsilon value to prevent numerical problems caused by taking the logarithm of zero.

```text
Loss = -mean(sum(y_true × log(y_pred)))
```

A lower loss indicates that the predictions are becoming closer to the actual labels.

---

### 3.5 Optimisation

The network is trained using **Mini-Batch Gradient Descent**.

The training configuration is:

```text
Learning Rate = 0.01
Epochs = 20
Batch Size = 64
```

During each epoch:

1. The training data is shuffled.
2. The data is divided into mini-batches of size 64.
3. A forward pass is performed.
4. Cross-entropy loss is calculated.
5. Backpropagation calculates the gradients.
6. Weights and biases are updated using gradient descent.
7. Training loss and accuracy are recorded.

The parameter update is performed using:

```text
W = W - learning_rate × dW
```

and similarly for the biases.

---

## 4. Implementation

The neural network is implemented using **Python and NumPy** without using high-level deep learning frameworks such as TensorFlow, Keras, or PyTorch.

The implementation includes the following components:

* ReLU activation function
* ReLU derivative
* Softmax activation function
* One-hot encoding
* Cross-Entropy Loss
* Weight and bias initialization
* Forward propagation
* Backpropagation
* Gradient Descent
* Mini-batch training
* Prediction
* Accuracy calculation
* Confusion matrix
* Model saving and loading

The trained parameters are saved in:

```text
mnist_neural_network_weights.npz
```

The saved weights and biases can be loaded later without retraining the model.

---

## 5. Results

The model was trained for **20 epochs** using a learning rate of **0.01** and a batch size of **64**.

### Training Progress

| Epoch |   Loss | Accuracy |
| ----: | -----: | -------: |
|     1 | 1.7696 |   72.80% |
|     5 | 0.5253 |   87.87% |
|    10 | 0.3792 |   90.14% |
|    15 | 0.3262 |   91.22% |
|    20 | 0.2933 |   91.98% |

### Final Performance

* **Final Training Loss:** 0.2933
* **Final Training Accuracy:** 91.98%
* **Test Accuracy:** 90.95%

The model achieved a test accuracy of **90.95%** on the 10,000 MNIST test images.

A confusion matrix was also generated to analyze the correct and incorrect predictions for each digit class.

---

## 6. Prediction Example

For one test image, the model produced the following result:

```text
Actual Digit: 7
Predicted Digit: 7
```

The model assigned a probability of:

```text
Digit 7: 99.33%
```

for this particular image.

The notebook also visualizes individual test images and compares their actual and predicted labels.

---

## 7. Model Saving and Loading

After training, the weights and biases are saved using NumPy's `.npz` format:

```text
mnist_neural_network_weights.npz
```

The saved parameters include:

```text
W1
b1
W2
b2
```

These parameters can be loaded later to reconstruct the trained neural network and perform predictions without training the model again.

---

## 8. Tools and Libraries

* **Python**
* **NumPy** – numerical computations and neural network implementation
* **Pandas** – displaying prediction results
* **Matplotlib** – visualizing training accuracy, images, and confusion matrix
* **Scikit-learn** – used for obtaining the MNIST dataset

No built-in deep learning framework was used for implementing or training the neural network.

---

## 9. Conclusion

This assignment demonstrates the implementation of a feedforward neural network from scratch for handwritten digit classification using the MNIST dataset.

The implementation covers the complete basic training pipeline, including forward propagation, cross-entropy loss calculation, backpropagation, and mini-batch gradient descent.

The neural network achieved a **91.98% training accuracy** and **90.95% test accuracy**, demonstrating that a basic neural network can successfully perform handwritten digit classification when its core components are implemented manually.

---

## 10. Declaration

I hereby declare that this assignment, **“Neural Network Implementation from Scratch,”** has been implemented by me as an individual lab task. The neural network components were implemented from scratch in Python without using built-in deep learning libraries.

**Submitted by:**
**Dhanashri Abhijit Shah**
