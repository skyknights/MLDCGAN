# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

def save_sample_images(mri_input, ct_target, generated_ct, save_path):
    """保存样本图像进行可视化"""
    batch_size = mri_input.size(0)
    
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4*batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # MRI T1W
        axes[i, 0].imshow(mri_input[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('T1W')
        axes[i, 0].axis('off')
        
        # MRI T2W
        axes[i, 1].imshow(mri_input[i, 1].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('T2W')
        axes[i, 1].axis('off')
        
        # Mask
        axes[i, 2].imshow(mri_input[i, 2].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title('Mask')
        axes[i, 2].axis('off')
        
        # Target CT
        axes[i, 3].imshow(ct_target[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 3].set_title('Target CT')
        axes[i, 3].axis('off')
        
        # Generated CT
        axes[i, 4].imshow(generated_ct[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 4].set_title('Generated CT')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def denormalize_image(image):
    """将图像从[-1, 1]反归一化到[0, 1]"""
    return (image + 1.0) / 2.0

def calculate_metrics(generated, target):
    """计算图像质量指标"""
    generated = denormalize_image(generated)
    target = denormalize_image(target)
    
    generated_np = generated.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = generated_np.shape[0]
    ssim_scores = []
    psnr_scores = []
    mae_scores = []
    
    for i in range(batch_size):
        gen_img = generated_np[i, 0]
        tar_img = target_np[i, 0]
        
        # SSIM
        ssim_score = ssim(tar_img, gen_img, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        # PSNR
        psnr_score = psnr(tar_img, gen_img, data_range=1.0)
        psnr_scores.append(psnr_score)
        
        # MAE
        mae_score = np.mean(np.abs(gen_img - tar_img))
        mae_scores.append(mae_score)
    
    return {
        'SSIM': np.mean(ssim_scores),
        'PSNR': np.mean(psnr_scores),
        'MAE': np.mean(mae_scores)
    }

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer=None):
    """加载训练检查点"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
