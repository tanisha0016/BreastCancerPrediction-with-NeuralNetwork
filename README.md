# BreastCancerPrediction-with-NeuralNetwork
# Breast Cancer Classification

## Overview

This project aims to classify breast cancer cases using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to predict whether a tumor is malignant or benign based on various cell nuclei characteristics.

## Dataset

The dataset includes 569 instances with 30 numerical features describing cell nuclei characteristics and a target variable indicating tumor malignancy.

## Installation

To get started, you'll need Python and the following libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage
Here's a brief example of how to load and explore the dataset:
```bash
import numpy as np
import pandas as pd
import sklearn.datasets

# Load the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Create a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Display the first few rows
print(data_frame.head())
```

## Neural Network Classification
This project demonstrates how to use neural networks for classifying tumor malignancy. We use TensorFlow and Keras to build and train a neural network. Here’s a basic example:
```bash
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
```
## Contributing
Feel free to fork this repository and submit pull requests with improvements or fixes. For any issues, please open an issue on the repository page.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Dataset: Breast Cancer Wisconsin (Diagnostic) dataset, available from UCI Machine Learning Repository.
Original dataset creator: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian.
