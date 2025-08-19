# config.py
import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description='MLDCGAN for MRI-CT Synthesis')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--num_timesteps', type=int, default=4, help='Number of diffusion timesteps')
    parser.add_argument('--beta_1', type=float, default=1e-4, help='Beta 1 for noise schedule')
    parser.add_argument('--beta_T', type=float, default=0.02, help='Beta T for noise schedule')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--lambda_fm', type=float, default=10.0, help='Feature matching loss weight')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()