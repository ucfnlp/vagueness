#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np


# computes accuracy and f1 scores
def performance(predicted, truth, negative_label=0):
    predicted_flat = predicted.flatten()
    truth_flat = truth.flatten()
    assert(predicted_flat.shape == truth_flat.shape)
    TP = np.sum(predicted_flat[truth_flat == 1] == 1)
    FP = np.sum(predicted_flat[truth_flat == negative_label] == 1)
    TN = np.sum(predicted_flat[truth_flat == negative_label] == negative_label)
    FN = np.sum(predicted_flat[truth_flat == 1] == negative_label)
    accuracy = 1.0 * np.sum(predicted_flat == truth_flat) / truth_flat.size
    precision = 1.0 * TP / ((TP + FP) or 1)
    recall = 1.0 * TP / ((TP + FN) or 1)
    f1 = 2.0 * precision * recall / ((recall + precision) or 1)
#     sensitivity = 1.0 * TP / ((TP + FN)  or 1)
#     specificity = 1.0 * TN / ((TN + FP) or 1)
    return accuracy, precision, recall, f1