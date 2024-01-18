import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple feedforward neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=8)) # input layer with 8 features
model.add(Dense(32, activation='relu'))   # input layer with 32 features
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification

# display model summary
model.summary()