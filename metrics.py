#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np


# computes accuracy and f1 scores
def performance(predicted, truth):
    TP = np.sum(predicted[truth == 1] == 1)
    FP = np.sum(predicted[truth == 0] == 1)
    TN = np.sum(predicted[truth == 0] == 0)
    FN = np.sum(predicted[truth == 1] == 0)
    accuracy = 1.0 * np.sum(predicted == truth) / truth.size
    precision = 1.0 * TP / ((TP + FP) or 1)
    recall = 1.0 * TP / ((TP + FN) or 1)
    f1 = 2.0 * precision * recall / ((recall + precision) or 1)
#     sensitivity = 1.0 * TP / ((TP + FN)  or 1)
#     specificity = 1.0 * TN / ((TN + FP) or 1)
    return accuracy, precision, recall, f1