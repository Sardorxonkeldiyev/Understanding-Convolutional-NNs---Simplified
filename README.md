# Understanding-Convolutional-NNs---Simplified

# CIFAR-10 Image Classification using Convolutional Neural Network (CNN)

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. This implementation uses a subset of the dataset for training and testing.

## Dataset

The CIFAR-10 dataset contains the following classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is divided into 50,000 training images and 10,000 testing images.

## Model Architecture

The model is a sequential Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layer:** 32 filters, kernel size (3, 3), ReLU activation
2. **Max Pooling Layer:** Pool size (2, 2)
3. **Convolutional Layer:** 64 filters, kernel size (3, 3), ReLU activation
4. **Max Pooling Layer:** Pool size (2, 2)
5. **Flatten Layer**
6. **Dense Layer:** 128 units, ReLU activation
7. **Dense Layer:** 10 units, Softmax activation (for classification)

## Preprocessing

- **Data Normalization:** The pixel values of the images are scaled to the range [0, 1] by dividing by 255.
- **One-Hot Encoding:** The class labels are one-hot encoded to create binary matrices for each category.

## Training

The model is trained on 10% of the CIFAR-10 dataset to reduce computational complexity and training time. The subset consists of 5,000 training images and 1,000 testing images. The model is trained for 10 epochs with a batch size of 64.

## Usage

To run the model, execute the following Python script:

```python
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the fraction of the dataset to be used (10%)
subset_fraction = 0.1

# Reduce the training data
train_subset_size = int(len(x_train) * subset_fraction)
x_train_subset = x_train[:train_subset_size]
y_train_subset = y_train[:train_subset_size]

# Reduce the test data
test_subset_size = int(len(x_test) * subset_fraction)
x_test_subset = x_test[:test_subset_size]
y_test_subset = y_test[:test_subset_size]

# Normalize the data
x_train_subset, x_test_subset = x_train_subset / 255.0, x_test_subset / 255.0

# One-hot encode the labels
y_train_subset = to_categorical(y_train_subset, num_classes=10)
y_test_subset = to_categorical(y_test_subset, num_classes=10)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_subset, y_train_subset, epochs=10, batch_size=64, verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_subset, y_test_subset, verbose=2)

print('\nTest accuracy:', test_acc)



Results
The model's performance is evaluated on the test subset. The test accuracy is printed after the evaluation.

Requirements
TensorFlow
NumPy
To install the required libraries, run:

sh
Copy code
pip install tensorflow numpy
Conclusion
This project demonstrates a basic implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model architecture can be further tuned and improved by experimenting with different hyperparameters, layers, and techniques.
