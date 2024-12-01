from models.skin_cancer_classifier import SkinCancerClassifier
from models.helpers import init_weights
from datasets.isic_2024 import ISIC2024Dataset, ISIC2024Split
from transforms.transforms import transform_isic_2024
from losses.losses import FocalLoss
from metrics.metrics import EvalMetrics
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import datetime
from typing import Union
from pathlib import Path
from typing import Tuple

def train_loop(seed: int, num_epochs: int, batch_size: int, lr: float, wd: float, input_size: int, unet_weights_path: str, scc_weights_path: Union[None,str], data_root: str, output_path: str, lr_scheduler_factor: float, lr_scheduler_patience: int, num_workers: int, pin_memory: bool=True, split_dataset: bool=True, split_folder: str=None, split_ratio: Tuple=(0.8, 0.1, 0.1), num_val_points: int=10, focal_loss_alpha: float=1, focal_loss_gamma: float=2):

    timestamp = datetime.datetime.now()

    writer = SummaryWriter()
    writer.add_text("Start timestamp", f"Start timestamp: {timestamp}")

    if os.path.exists(output_path) is False:
        os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_weights = torch.load(unet_weights_path, map_location=device)['model_state_dict'] 
    model = SkinCancerClassifier(unet_weights=unet_weights).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, cooldown=0, verbose=True)
    
    epoch_0 = 0

    if scc_weights_path:
        if os.path.exists(scc_weights_path):
            try:
                checkpoint = torch.load(scc_weights_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_0 = checkpoint['epoch']
                print(f"Loaded checkpoint at {scc_weights_path}")
            except Exception as e:
                model.apply(init_weights)
                print(f"ERROR: Attempt at loading weights from {scc_weights_path} threw an exception {e}.\nModel was xavier initialised instead!!!")    
        else:
            model.apply(init_weights)
            print("Specified weights path does not exist, model was xavier initialised")

    split_settings = ISIC2024Split(seed, data_root, split_ratio=split_ratio, writer=writer)

    Path(split_folder).mkdir(parents=True, exist_ok=True)
    if split_dataset:
        split_settings.split()
        split_settings.save(split_folder)
    else:
        split_settings.load(split_folder)    
    
    criterion = FocalLoss(focal_loss_alpha, focal_loss_gamma, reduction='mean').to(device)

    train_dataset = ISIC2024Dataset(split=split_settings, transform=transform_isic_2024(input_size), mode='train', writer=writer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = ISIC2024Dataset(split=split_settings, transform=transform_isic_2024(input_size), mode='val', writer=writer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    iter_epoch = len(train_loader)
    val_every_iters = iter_epoch // num_val_points
    
    for epoch in range(epoch_0, num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        last_lr = optimizer.param_groups[0]['lr']  
        model.train()

        for idx, (images, gradings) in enumerate(tqdm(train_loader)):
            iter = idx +  iter_epoch * epoch
            images = images.to(device)
            gradings = gradings.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gradings.to(torch.float32))
            writer.add_scalar("train. loss (iter)", loss, iter)
            writer.add_scalar("lr (iter)", last_lr, iter)
            loss.backward()
            optimizer.step()
            if (iter + 1) % val_every_iters == 0:
                print(f"Evaluation at iter {iter} in progress...")
                model.eval()
                val_loss = 0
                best_val_loss = torch.inf
                pAUC_80tpr = 0
                test_gradings = torch.empty((0,2)).to(device)
                test_outputs = torch.empty((0,2)).to(device)
                with torch.no_grad():
                    for val_idx, (images, gradings) in tqdm(enumerate(val_loader)):
                        images = images.to(device)
                        outputs = model(images)
                        gradings = gradings.to(device)
                        test_gradings = torch.cat((test_gradings, gradings), dim = 0)
                        test_outputs = torch.cat((test_outputs, outputs), dim = 0)
                        val_loss += criterion(outputs, gradings.to(torch.float32))
                test_gradings = test_gradings.cpu()
                test_outputs = test_outputs.cpu()
                eval_metrics = EvalMetrics(gt_target = test_gradings, f1_threshold=0.5, writer=writer)
                pr_metrics = eval_metrics.pr_metrics(test_outputs)
                pAUC_80tpr = pr_metrics['pAUC_80tpr']
                val_loss = val_loss / (val_idx + 1)
                writer.add_scalar("pAUC_80tpr (iter)", pAUC_80tpr, iter)
                writer.add_scalar("val. loss (iter)", val_loss, iter)
                print(f"pAUC_80tpr (iter={iter}) = {pAUC_80tpr}")
                print(f"val. loss (iter={iter}) = {val_loss}")
                print(f"LR = {last_lr}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss,
                                'val_loss': val_loss,
                                'pAUC_80tpr': pAUC_80tpr
                                }, os.path.join(output_path, f'best-scc-model_{timestamp}.pth'))
                scheduler.step(val_loss)
                model.train()
        writer.add_scalar("train. loss (epoch)", loss, epoch)
    writer.flush()