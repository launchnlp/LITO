import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_figure(epochs, info_dict, mode, dataset, model):
    for cl, acc in info_dict.items():
        c_acc = acc

        # Plotting per-class accuracy
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, acc, label=f'Class {cl} Accuracy', marker='o')

        plt.title('Per-Class Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/{dataset}/{model}/{cl}_{mode}.png')

def f1_calc(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def conf_matrix_calc(labels, predictions):
    tp = torch.sum((labels == 1) & (predictions == 1)).item()
    fp = torch.sum((labels == 0) & (predictions == 1)).item()
    tn = torch.sum((labels == 0) & (predictions == 0)).item()
    fn = torch.sum((labels == 1) & (predictions == 0)).item()
    return tp, fp, tn, fn


def balanced_accuracy_calc(tp, fp, tn, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy

if __name__ == "__main__":
    charts()
