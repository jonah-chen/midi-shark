import torch

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    """
    true_positives = torch.sum(y_true & y_pred)
    predicted_positives = torch.sum(y_pred)
    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    """
    true_positives = torch.sum(y_true & y_pred)
    possible_positives = torch.sum(y_true)
    recall = true_positives / (possible_positives + 1e-7)
    return recall

def f1_score(y_true, y_pred):
    """F1 score.
    Only computes a batch-wise average of f1 score.
    Computes the f1 score, a metric for multi-label classification of
    """
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))