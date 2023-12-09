# House-price-prediction-using-Non-linear-Polynomial-Regression-using-Gradient-Descent
This code that I have created is to predict the House price in Australia by training the Regression using given Housing Data

## Overview

Welcome to the House Price Prediction project! This project focuses on predicting house prices using a non-linear regression model implemented from scratch without relying on inbuilt libraries. The primary algorithm used is gradient descent for updating the coefficients of the regression model.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Introduction

The goal of this project is to develop an accurate house price prediction model using non-linear regression. Unlike traditional linear models, non-linear regression allows capturing more complex relationships between house features and prices, offering a more nuanced prediction.

## Features

The model takes into consideration various features to predict house prices, including square footage, number of bedrooms, and location. The non-linear regression model implemented in this project aims to account for intricate relationships in the dataset.

## Dataset

The dataset used for training and testing the model comprises information on house features and corresponding prices. The dataset has been preprocessed to handle any missing values and outliers.

## Implementation Details

The non-linear regression model is implemented without using inbuilt libraries. The coefficients of the model are updated using the gradient descent algorithm. This approach enables fine-tuning the model's parameters for improved accuracy.

## Usage

To train the model and make predictions, follow these steps:

1. **Initialization**: Create an instance of the `NonLinearRegression` class.
2. **Training**: Use the `fit` method with your training data.
3. **Prediction**: Utilize the `predict` method to make predictions on new data.
4. **Analysis**: Display coefficients, plot results, and evaluate accuracy.


## Results

The model's accuracy and performance metrics, including training and testing accuracy, are displayed during the training process. Additionally, the final coefficients and cost values over iterations are visualized.

## Contributing

Contributions to the project are welcome! Feel free to submit issues or pull requests.

## Author

- **Naga Koushik Sajja**
  - Phone: 9392888109
  - Email: nagakoushik24@gmail.com


Example:

```python
# Import the necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Instantiate the NonLinearRegression class
regression_model = NonLinearRegression(alpha=0.01)

# Train the model with your data
regression_model.fit(x_train, y_train, z_train, w_train)

# Make predictions
predictions = regression_model.predict(x_new, z_new, w_new)
print(predictions)

# Display coefficients and plot results
regression_model.display_coefficients()
regression_model.plot_results()
