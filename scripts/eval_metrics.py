
import torch

def recall_at_all(logits, y_true, agg='sum'):
    """
    Modification of recall @ K to set K = # of side effects for that drug
    Using agg='sum' to be able to later normalize when adding all together
    """

    y_true = y_true.float()
    batch_size, num_classes = y_true.shape

    _, sorted_indices = torch.sort(logits, dim=1, descending=True)

    recalls = []
    for i in range(batch_size):

        true_labels = y_true[i]
        num_pos = int(true_labels.sum().item())

        topk_idx = sorted_indices[i, :num_pos]
        
        true_pos_in_topk = true_labels[topk_idx].sum().item()
        recall_i = true_pos_in_topk / num_pos
        recalls.append(recall_i)

    if agg == 'sum':
        return sum(recalls)
    
    elif agg == 'mean':
        return sum(recalls) / len(recalls)
    
    else:
        return 