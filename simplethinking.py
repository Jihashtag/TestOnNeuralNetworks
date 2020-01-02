#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.weights = np.random.rand(self.inputs.shape[1], 1)

        self.last_epoch = 0
        self.error_history = []
        self.epoch_list = []

        if len(inputs):
            self.train()

    def update_data(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.weights = np.random.rand(self.inputs.shape[1], 1)
        self.train()

    def sigmoid(self, x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def get_questions_weights(self, questions):
        for i in range(len(questions)):
            print(questions[i], round(self.weights[i][0], 2))

    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = (0.5 * self.error) * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch + self.last_epoch)
        self.last_epoch += epoch

    # function to predict output on new and unseen input data
    def predict(self, new_input, addInfos=False):
        if not isinstance(new_input, np.ndarray):
            new_input = np.array(new_input)
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        if addInfos:
            return prediction, prediction
        return prediction
