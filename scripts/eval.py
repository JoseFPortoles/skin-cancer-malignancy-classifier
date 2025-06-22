import torch
from metrics.metrics import EvalMetrics

class Evaluator:
    def __init__(self, model_path='models/model.pth', checkpoint_path='weights/unet/checkpoint.pth'):
        self.model = self.load_model(model_path)
        self.checkpoint = self.load_checkpoint(checkpoint_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()  # Set the model to evaluation mode
        return model

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint

    def evaluate(self, data_loader):
        # Recoger los targets de todo el dataset
        ground_truth = []
        for batch in data_loader:
            inputs, targets = batch
            ground_truth.extend(targets)  # Acumular los targets en una lista
            outputs = self.model(inputs)
            # Aquí puedes calcular métricas usando eval_metrics

        # Usar get_target para procesar la ground truth
        # Instanciar EvalMetrics
        eval_metrics = EvalMetrics(gt_target=torch.tensor(ground_truth, dtype=torch.float32))
        # Calcular métricas de precisión y recall
        metrics = eval_metrics.pr_metrics(outputs)
        # Mostrar la curva ROC en TensorBoard
        #eval_metrics.display_roc_curve_in_tensorboard()
        # Mostrar la curva PR en matplotlib
        eval_metrics.display_pr_curve_in_pyplot()        
        return metrics