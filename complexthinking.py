#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

        dataShape = self.inputs.shape[1]
        self.weights1 = np.random.rand(dataShape, 3)
        self.weights2 = np.random.rand(3, 1)

        self.last_epoch = 0
        self.error_history = []
        self.epoch_list = []

        if len(inputs):
            self.train()

    def update_data(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

        dataShape = self.inputs.shape[1]
        self.weights1 = np.random.rand(dataShape, 3)
        self.weights2 = np.random.rand(3, 1)

        self.train()

    def sigmoid(self, x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def get_questions_weights(self, questions):
        for i in range(len(questions)):
            print(questions[i], list(round(x, 2) for x in self.weights1[i]))
        print(" ====== ====== ")
        for w in self.weights2:
            print(w[0], end=" ")
        print("\n---   ---   ---")

    def feed_forward(self):
        self.hidden1 = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights2))

    def backpropagation(self):
        d_weights2 = np.dot(self.hidden1.T, (2 * (self.outputs - self.hidden2) * self.sigmoid(self.hidden2, True)))
        d_weights1 = np.dot(self.inputs.T, (np.dot(2 * (self.outputs - self.hidden2) * self.sigmoid(self.hidden2, True), self.weights2.T) * self.sigmoid(self.hidden1, True)))

        self.error = self.outputs - self.hidden2

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, epochs=50000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch + self.last_epoch)
        self.last_epoch += epoch

    # function to predict output on new and unseen input data
    def predict(self, new_input, addInfos=False):
        if not isinstance(new_input, np.ndarray):
            new_input = np.array(new_input)
        firstLayer = self.sigmoid(np.dot(new_input, self.weights1))
        prediction = self.sigmoid(np.dot(firstLayer, self.weights2))
        if addInfos:
            return prediction, firstLayer
        return prediction
