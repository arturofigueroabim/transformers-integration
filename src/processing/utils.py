from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def compute_class_weights(labels):
    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(labels), y = labels)
    dic_class_weights = dict(zip(np.unique(labels), class_weights))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights.to(device), dic_class_weights
