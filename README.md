# Transport Mode Classifier using Smartphone Sensor Data

This project focuses on classifying the mode of transportation (e.g., **car**, **bus**, **train**, or **other**) using smartphone sensor data, primarily from the **accelerometer** and **gyroscope**. A supervised machine learning approach was used to train and evaluate a model capable of identifying movement patterns corresponding to different transport modes.

## Project Overview

- **Goal**: Automatically predict the user’s mode of transport using time-series sensor data.
- **Dataset**: Collected from a smartphone with labeled intervals (car, bus, train, other).
- **Model Used**: Random Forest Classifier (others like SVM and k-NN tested during prototyping).
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Instructions 
1. To install all dependencies, run `pip install -r requirements.txt`

2. To predict a trip, you must include the path to it in `predict_mode.py` samples are provided in test-set folder.

3. If you want to see the model testing results, run `train_model.py` or `tune_model.py` 
   Their difference is stated in tune_model.py file.
