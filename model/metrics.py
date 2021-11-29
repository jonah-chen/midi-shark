import torch

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    """
    true_positives = torch.sum(y_true & y_pred)
    predicted_positives = torch.sum(y_pred)
    precision = true_positives / (predicted_positives + 1e-7)
    return precision.item()

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    """
    true_positives = torch.sum(y_true & y_pred)
    possible_positives = torch.sum(y_true)
    recall = true_positives / (possible_positives + 1e-7)
    return recall.item()

def f1_score(y_true, y_pred):
    """F1 score.
    Only computes a batch-wise average of f1 score.
    Computes the f1 score, a metric for multi-label classification of
    """
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-7)).item()

class modified_mse:
    def __call__(self, y_true, y_pred, onsets_true):
        """Modified mean squared error.
        Only computes a batch-wise average of mse.
        Computes the mean squared error between the labels and predictions.
        """
        total_onsets = onsets_true.sum()
        return (onsets_true*(y_true-y_pred)**2).sum() / total_onsets if total_onsets else 0

def err(y_true, y_pred, onsets_true, max_err=0.01):
    total_onsets = onsets_true.sum()
    return (onsets_true* ( (y_true-y_pred)**2 < max_err ) ).sum() / total_onsets if total_onsets else 0