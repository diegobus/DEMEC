import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(logits, y_true, k=50):
    """
    Precision@K: What percentage of top-K predictions are correct?
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        k: Number of top predictions to consider
        
    Returns:
        Average precision@K across the batch
    """
    y_true = y_true.float()
    _, top_k_indices = torch.topk(logits, k, dim=1)
    
    precisions = []
    for i in range(len(y_true)):
        true_labels = y_true[i]
        top_k_preds = top_k_indices[i]
        
        # How many of top-K are actually positive?
        hits = true_labels[top_k_preds].sum().item()
        precision = hits / k
        precisions.append(precision)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


def recall_at_k(logits, y_true, k=50):
    """
    Recall@K: What percentage of true positives are in top-K predictions?
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        k: Number of top predictions to consider
        
    Returns:
        Average recall@K across the batch
    """
    y_true = y_true.float()
    _, top_k_indices = torch.topk(logits, k, dim=1)
    
    recalls = []
    for i in range(len(y_true)):
        true_labels = y_true[i]
        num_pos = true_labels.sum().item()
        
        if num_pos == 0:
            continue
            
        top_k_preds = top_k_indices[i]
        hits = true_labels[top_k_preds].sum().item()
        recall = hits / num_pos
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def f1_at_k(logits, y_true, k=50):
    """
    F1@K: Harmonic mean of precision and recall at K.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        k: Number of top predictions to consider
        
    Returns:
        F1 score at K
    """
    prec = precision_at_k(logits, y_true, k)
    rec = recall_at_k(logits, y_true, k)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def mean_average_precision(logits, y_true):
    """
    Mean Average Precision (mAP): Area under the precision-recall curve.
    Gold standard metric for ranking tasks.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        
    Returns:
        Mean average precision across the batch
    """
    logits_np = logits.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    aps = []
    for i in range(len(y_true)):
        if y_true_np[i].sum() > 0:  # Skip samples with no positives
            try:
                ap = average_precision_score(y_true_np[i], logits_np[i])
                aps.append(ap)
            except:
                continue
    
    return sum(aps) / len(aps) if aps else 0.0


def compute_auroc(logits, y_true, average='micro'):
    """
    Area Under ROC Curve: Overall classification quality.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        average: 'micro' (flatten all) or 'macro' (per-class then average)
        
    Returns:
        AUROC score
    """
    logits_np = logits.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    if average == 'micro':
        # Flatten everything
        logits_flat = logits_np.ravel()
        y_true_flat = y_true_np.ravel()
        try:
            return roc_auc_score(y_true_flat, logits_flat)
        except:
            return 0.0
    else:
        # Per-class AUROC
        aurocs = []
        for j in range(y_true_np.shape[1]):
            if len(np.unique(y_true_np[:, j])) > 1:  # Need both classes
                try:
                    auroc = roc_auc_score(y_true_np[:, j], logits_np[:, j])
                    aurocs.append(auroc)
                except:
                    continue
        return sum(aurocs) / len(aurocs) if aurocs else 0.0


def comprehensive_metrics(logits, y_true, k_values=[10, 50, 100]):
    """
    Compute comprehensive evaluation metrics for multi-label classification.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        y_true: True labels (batch_size, num_classes)
        k_values: List of K values for Precision/Recall/F1@K
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Mean Average Precision (primary metric)
    metrics['mAP'] = mean_average_precision(logits, y_true)
    
    # Precision/Recall/F1 at various K values
    for k in k_values:
        metrics[f'P@{k}'] = precision_at_k(logits, y_true, k)
        metrics[f'R@{k}'] = recall_at_k(logits, y_true, k)
        metrics[f'F1@{k}'] = f1_at_k(logits, y_true, k)
    
    # Overall classification quality
    metrics['AUROC'] = compute_auroc(logits, y_true, average='micro')
    
    return metrics


# Backward compatibility: keep old metric but mark as deprecated
def recall_at_all(logits, y_true, agg='sum'):
    """
    DEPRECATED: Use comprehensive_metrics() instead.
    
    Old metric: Recall@K where K = number of true positives per sample.
    This metric is optimistic and not comparable across samples with
    different numbers of positives.
    """
    y_true = y_true.float()
    batch_size, num_classes = y_true.shape

    _, sorted_indices = torch.sort(logits, dim=1, descending=True)

    recalls = []
    for i in range(batch_size):
        true_labels = y_true[i]
        num_pos = int(true_labels.sum().item())
        
        if num_pos == 0:
            continue

        topk_idx = sorted_indices[i, :num_pos]
        true_pos_in_topk = true_labels[topk_idx].sum().item()
        recall_i = true_pos_in_topk / num_pos
        recalls.append(recall_i)

    if agg == 'sum':
        return sum(recalls)
    elif agg == 'mean':
        return sum(recalls) / len(recalls) if recalls else 0.0
    else:
        return 0.0