
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import os
import numpy as np

def compute_metrics_ce(outputs, labels, threshold=0.5):
    """
    Compute metrics based on CrossEntropy Loss outputs.

    Args:
        outputs: (N, 2) model logits.
        labels: (N,) ground-truth labels (0 or 1).
        threshold: decision threshold applied to the probability of class 1.

    Returns:
        accuracy, precision, recall, f1, auc, tn, fp, fn, tp
    """

    outputs = np.array(outputs)
    labels = np.array(labels)

    # Softmax to get probability of class 1
    probs = torch.softmax(torch.tensor(outputs), dim=1)[:, 1].numpy()

    # Binarize with the given threshold
    bin_preds = (probs > threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds, zero_division=0)
    recall = recall_score(labels, bin_preds, zero_division=0)
    f1 = f1_score(labels, bin_preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, bin_preds, labels=[0, 1]).ravel()

    return accuracy, precision, recall, f1, auc, tn, fp, fn, tp



def compute_metrics_bce(preds, labels, threshold=0.5):
    """
    Compute metrics based on Binary Cross-Entropy Loss outputs.

    Args:
        preds: (N,) raw logits for the positive class.
        labels: (N,) ground-truth labels (0 or 1).
        threshold: decision threshold after sigmoid.

    Returns:
        accuracy, precision, recall, f1, auc, tn, fp, fn, tp
    """
    preds = np.array(preds)
    probs = torch.sigmoid(torch.tensor(preds, dtype=torch.float32)).numpy()
    print("probs:", probs)
    bin_preds = (probs > threshold).astype(int)

    accuracy = accuracy_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds, zero_division=0)
    recall = recall_score(labels, bin_preds, zero_division=0)
    f1 = f1_score(labels, bin_preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, bin_preds, labels=[0, 1]).ravel()

    return accuracy, precision, recall, f1, auc, tn, fp, fn, tp

def save_metrics_to_csv(stage, backbone, model_weights, dataset_name, classification_loss, metrics, filename):
    accuracy, precision, recall, f1, auc, tn, fp, fn, tp = metrics

    row = {
        "Stage": stage,
        "Backbone": backbone,
        "Model Weights": model_weights,
        "Dataset": dataset_name,
        "Classificaion Loss": classification_loss.item(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode="a", index=False, header=not file_exists)
