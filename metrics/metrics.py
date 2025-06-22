import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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
        # Add roc curve to class fields
        self.metrics['roc_curve'] = (fpr, tpr, roc_thresholds)

        return {'f1_score': f1, 
                'precision': precision, 
                'recall': recall, 
                'pr_thresholds': pr_thresholds, 
                'roc_thresholds': roc_thresholds,
                'AUC': auc_pr,
                'pAUC_80tpr': pauc_80tpr}
    
    def display_roc_curve_in_tensorboard(self):
        """Displays the ROC curve in Tensorboard
        """
        if 'roc_curve' not in self.metrics:
            raise ValueError("ROC curve data not available. Run pr_metrics first.")
        
        fpr, tpr, roc_thresholds = self.metrics['roc_curve']
        self.writer.add_pr_curve('ROC Curve', self.gt_malignant, fpr, tpr, roc_thresholds)

    def display_pr_curve_in_pyplot(self):
        """Displays the Precision-Recall curve using matplotlib
        """
        if 'roc_curve' not in self.metrics:
            raise ValueError("ROC curve data not available. Run pr_metrics first.")
        
        fpr, tpr, roc_thresholds = self.metrics['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()
        plt.show()