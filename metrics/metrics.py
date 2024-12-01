import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class EvalMetrics:
    """Wraps evaluation metrics and functionality
    """    
    def __init__(self, gt_target: torch.Tensor, f1_threshold: float=0.5, writer: SummaryWriter=None):
        """Initialises EvalMetrics

        Args:
            gt_target (torch.Tensor): ground truth labels tensor for [benign, malignant] categories, e.g: torch.Tensor([[0,1], [0,1], [1,0]]) => [malignant, malignant, benign] 
            f1_threshold: Threshold for converting probabilities to labels before computing F1-score
            writer: Tensorboard writer for logging
        """        
        self.gt_malignant = gt_target[:,1]
        self.metrics = dict()
        self.writer = writer
        self.f1_threshold = f1_threshold

    def pr_metrics(self, pred: torch.Tensor):
        """Computes, logs and returns different precision-recall based metrics

        Args:
            pred (torch.Tensor): 2-D tensor with the predicted probabilities [p_benign, p_malignant], e.g: [[0.505, 0.495], [0.300, 0.700], ...]
        """
        prob_malignant = pred[:, 1]
        labels_malignant = torch.where(prob_malignant > self.f1_threshold, 1, 0)
        
        f1 = f1_score(self.gt_malignant, labels_malignant)
        
        precision, recall, pr_thresholds = precision_recall_curve(self.gt_malignant, prob_malignant)
        auc_pr = auc(recall, precision)
        
        fpr, tpr, roc_thresholds = roc_curve(self.gt_malignant, prob_malignant)
        fpr_partial80 = fpr[tpr >= 0.80]
        tpr_partial80 = tpr[tpr >= 0.80] - 0.80
        # pAUC above 80% TPR
        pauc_80tpr = auc(fpr_partial80, tpr_partial80)

        return {'f1_score': f1, 
                'precision': precision, 
                'recall': recall, 
                'pr_thresholds': pr_thresholds, 
                'roc_thresholds': roc_thresholds,
                'AUC': auc_pr,
                'pAUC_80tpr': pauc_80tpr}