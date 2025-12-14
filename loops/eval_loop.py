from models.skin_cancer_classifier import SkinCancerClassifier
from models.helpers import init_weights
from datasets.isic_2024 import ISIC2024Dataset, ISIC2024Split
from transforms.transforms import transform_isic_2024
from metrics.metrics import EvalMetrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import datetime
from typing import Union
from pathlib import Path
from typing import Tuple

def eval_loop(seed: int, batch_size: int, input_size: int, unet_weights_path: str, scc_weights_path: Union[None,str], data_root: str, output_path: str, num_workers: int, pin_memory: bool=True, split_dataset: bool=True, split_folder: str=None, split_ratio: Tuple=(0.8, 0.1, 0.1)):

    timestamp = datetime.datetime.now()

    writer = SummaryWriter()
    writer.add_text("Start timestamp", f"Start timestamp: {timestamp}")

    if os.path.exists(output_path) is False:
        os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_weights = torch.load(unet_weights_path, map_location=device)['model_state_dict'] 
    model = SkinCancerClassifier(unet_weights=unet_weights).to(device)

    if scc_weights_path:
        if os.path.exists(scc_weights_path):
            try:
                checkpoint = torch.load(scc_weights_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"ERROR: Attempt at loading weights from {scc_weights_path} threw an exception {e}.")    
        else:
            print(f"Specified weights path {scc_weights_path} does not exist.")

    split_settings = ISIC2024Split(seed, data_root, split_ratio=split_ratio, writer=writer)

    Path(split_folder).mkdir(parents=True, exist_ok=True)
    if split_dataset:
        split_settings.split()
        split_settings.save(split_folder)
    else:
        split_settings.load(split_folder)    
    

    eval_dataset = ISIC2024Dataset(split=split_settings, transform=transform_isic_2024(input_size), mode='val', writer=writer)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    model.eval()
    pAUC_80tpr = 0
    eval_gradings = torch.empty((0,2)).to(device)
    eval_outputs = torch.empty((0,2)).to(device)
    with torch.no_grad():
        for eval_idx, (images, gradings) in tqdm(enumerate(eval_loader)):
            images = images.to(device)
            outputs = model(images)
            gradings = gradings.to(device)
            eval_gradings = torch.cat((eval_gradings, gradings), dim = 0)
            eval_outputs = torch.cat((eval_outputs, outputs), dim = 0)
    eval_gradings = eval_gradings.cpu()
    eval_outputs = eval_outputs.cpu()
    eval_metrics = EvalMetrics(gt_target = eval_gradings, f1_threshold=0.5, writer=writer)
    pr_metrics = eval_metrics.pr_metrics(eval_outputs)
    pAUC_80tpr = pr_metrics['pAUC_80tpr']
    eval_loss = eval_loss / (eval_idx + 1)
    writer.add_scalar("pAUC_80tpr (iter)", pAUC_80tpr, iter)
    writer.add_scalar("val. loss (iter)", eval_loss, iter)
    print(f"pAUC_80tpr (iter={iter}) = {pAUC_80tpr}")
    print(f"eval. loss (iter={iter}) = {eval_loss}")
    writer.flush()