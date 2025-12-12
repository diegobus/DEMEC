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


def hamming_distance(logits, y_true, threshold=0.5):
    """
    Hamming distance: Number of bit positions that differ.
    Lower is better (0 = perfect match).
    
    Args:
        logits: Model predictions (batch_size, num_bits)
        y_true: True labels (batch_size, num_bits)
        threshold: Threshold for converting logits to binary
        
    Returns:
        Average Hamming distance (normalized by num_bits)
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    y_true = y_true.float()
    
    # Count differing bits per sample
    diff = (preds != y_true).float().sum(dim=1)
    num_bits = y_true.shape[1]
    
    # Normalize by number of bits
    return (diff / num_bits).mean().item()


def tanimoto_similarity(logits, y_true, threshold=0.5):
    """
    Tanimoto similarity (Jaccard index) for binary fingerprints.
    Higher is better (1.0 = perfect match).
    
    Args:
        logits: Model predictions (batch_size, num_bits)
        y_true: True labels (batch_size, num_bits)
        threshold: Threshold for converting logits to binary
        
    Returns:
        Average Tanimoto similarity
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    y_true = y_true.float()
    
    # Tanimoto = intersection / union
    intersection = (preds * y_true).sum(dim=1)
    union = (preds + y_true).clamp(max=1).sum(dim=1)
    
    # Avoid division by zero
    tanimoto = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    
    return tanimoto.mean().item()


def bit_accuracy(logits, y_true, threshold=0.5):
    """
    Percentage of bits predicted correctly.
    
    Args:
        logits: Model predictions (batch_size, num_bits)
        y_true: True labels (batch_size, num_bits)
        threshold: Threshold for converting logits to binary
        
    Returns:
        Bit-wise accuracy (0 to 1)
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    y_true = y_true.float()
    
    correct = (preds == y_true).float()
    return correct.mean().item()


def regression_metrics(preds, y_true):
    """
    Compute regression metrics (MSE, MAE, R²).
    
    Args:
        preds: Model predictions (batch_size, num_properties)
        y_true: True values (batch_size, num_properties)
        
    Returns:
        Dictionary of metrics
    """
    preds_np = preds.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    # MSE
    mse = ((preds_np - y_true_np) ** 2).mean()
    
    # MAE
    mae = np.abs(preds_np - y_true_np).mean()
    
    # R^2
    ss_res = ((y_true_np - preds_np) ** 2).sum()
    ss_tot = ((y_true_np - y_true_np.mean(axis=0)) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Per-property MAE
    per_property_mae = np.abs(preds_np - y_true_np).mean(axis=0)
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'R2': float(r2),
        'per_property_MAE': per_property_mae.tolist()
    }


def comprehensive_metrics(logits, y_true, k_values=[10, 50, 100], task_type='classification'):
    """
    Compute comprehensive evaluation metrics based on task type.
    
    Args:
        logits: Model predictions (batch_size, num_outputs)
        y_true: True labels (batch_size, num_outputs)
        k_values: List of K values for Precision/Recall/F1@K (classification only)
        task_type: 'classification', 'fingerprint', or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == 'regression':
        # Regression metrics (molprops)
        return regression_metrics(logits, y_true)
    
    elif task_type == 'fingerprint':
        # Binary fingerprint metrics (MACCS)
        metrics['Hamming'] = hamming_distance(logits, y_true)
        metrics['Tanimoto'] = tanimoto_similarity(logits, y_true)
        metrics['Bit_Acc'] = bit_accuracy(logits, y_true)
        metrics['AUROC'] = compute_auroc(logits, y_true, average='micro')
        
        # Also include mAP for comparison
        metrics['mAP'] = mean_average_precision(logits, y_true)
        
    else:
        # Multi-label classification metrics (side_effects, ATC)
        # Mean Average Precision (primary metric)
        metrics['mAP'] = mean_average_precision(logits, y_true)
        
        # Precision at small K values (more relevant for sparse labels)
        for k in [1, 5, 10]:
            metrics[f'P@{k}'] = precision_at_k(logits, y_true, k)
        
        # Overall classification quality
        metrics['AUROC'] = compute_auroc(logits, y_true, average='micro')
        
        # Top-1 accuracy (for very sparse labels like ATC)
        _, top1_idx = torch.topk(logits, 1, dim=1)
        top1_correct = 0
        for i in range(len(y_true)):
            if y_true[i, top1_idx[i, 0]] > 0:
                top1_correct += 1
        metrics['Top1_Acc'] = top1_correct / len(y_true)
    
    return metrics


# Backward compatibility (deprecated metric)
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