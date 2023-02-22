#!/usr/bin/env python
# coding: utf-8
# author: Yuanchen Bei

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, layer, batch_norm=True, dropout=0.5):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Dice(nn.Module):

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x + 1e-8)

        return x.mul(p) + self.alpha * x.mul(1 - p)
