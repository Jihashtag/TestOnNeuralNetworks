#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    weights_config = [3]
    hiddens = []

    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

        dataShape = self.inputs.shape[1]
        weights_config = [dataShape, *self.weights_config, 1]
        self.weights = []
        for i in range(len(weights_config) - 1):
            self.weights.append(np.random.rand(weights_config[i], weights_config[i + 1]))

        self.last_epoch = 0
        self.error_history = []
        self.epoch_list = []

        if len(inputs):
            self.train()

    def update_data(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

        dataShape = self.inputs.shape[1]
        weights_config = [dataShape, *self.weights_config, 1]
        self.weights = []
        for i in range(len(weights_config) - 1):
            self.weights.append(np.random.rand(weights_config[i], weights_config[i + 1]))

        self.train()

    def sigmoid(self, x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def get_questions_weights(self, questions):
        # Find a way to make it "readable" and print all weights
        for i in range(len(questions)):
            print(questions[i], list(round(x, 2) for x in self.weights[0][i]))
        print(" ====== ====== ")
        for w in self.weights[-1]:
            print(w[0], end=" ")
        print("\n---   ---   ---")

    def feed_forward(self):
        self.hiddens = []
        lhidden = self.inputs
        for weight in self.weights:
            lhidden = self.sigmoid(np.dot(lhidden, weight))
            self.hiddens.append(lhidden)

    def backpropagation(self):
        hiddens = self.hiddens.copy()
        hiddens.reverse()
        hiddens.append(self.inputs)
        d_weights = []
        for x in range(len(hiddens) - 1):
            if x == 0:
                lVal = (2 * (self.outputs - hiddens[0]) * self.sigmoid(hiddens[0], True))
            else:
                lVal = np.dot(lVal, self.weights[-x].T) * self.sigmoid(hiddens[x], True)
            d_weights.append(np.dot(hiddens[x + 1].T, lVal))

        self.error = self.outputs - hiddens[0]

        for x in range(len(d_weights)):
            self.weights[-(x + 1)] += d_weights[x]

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
        firstLayer = self.sigmoid(np.dot(new_input, self.weights[0]))
        lLayer = firstLayer
        for weight in self.weights[1:]:
            lLayer = self.sigmoid(np.dot(lLayer, weight))
        prediction = lLayer
        if addInfos:
            return prediction, firstLayer
        return prediction
