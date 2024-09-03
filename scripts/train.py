from models.skin_cancer_classifier import SkinCancerClassifier
from loops.train_loop import train_loop
from loops.seeds import seed_all 
import argparse

parser = argparse.ArgumentParser(prog='train.py', description='Train SCC with ISIC2024 data')

parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--input_size', type=int, default=224, help='Lateral size for the (square) input image.')
parser.add_argument('--unet_weights_path', type=str, default=None, help='U-Net weights file path. If =None then initialise net w/ Xavier function.')
parser.add_argument('--scc_weights_path', type=str, default=None, help='Saved complete network weights. If =None then initialise U-Net with xavier and ResNet with ImageNet weights.')
parser.add_argument('--data_root', type=str, help='Dataset root folder path.')
parser.add_argument('--output_path', type=str, default='./checkpoints', help='Folder where trained weights are saved.')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr reduction factor for the lr scheduler.')
parser.add_argument('--lr_scheduler_patience', type=int, default=3, help='lr scheduler patience.')
parser.add_argument('--num_workers', type=int, help='Num. workers for data loading.')
parser.add_argument('--pin_memory', action='store_true', help='Use pin memory in data loading')
parser.add_argument('--seed', type=int, help="Seed for all libraries")

args = parser.parse_args()


if __name__ == '__main__':
    seed_all(args.seed)
    model = SkinCancerClassifier()
    train_loop(args.seed, args.num_epochs, args.batch_size, args.lr, args.wd, args.input_size, args.unet_weights_path, args.scc_weights_path, args.data_root, args.output_path, args.lr_scheduler_factor, args.lr_scheduler_patience, args.num_workers, args.pin_memory)
    