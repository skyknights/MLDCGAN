# inference.py
import torch
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from generator import MLDCGANGenerator
from data_loader import MRICTDataset
from utils import save_sample_images, calculate_metrics, denormalize_image

class MLDCGANInference:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.generator = MLDCGANGenerator(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"模型已加载到设备: {self.device}")
    
    @torch.no_grad()
    def generate_ct(self, mri_input):
        """生成CT图像"""
        # 确保输入是正确的形状和设备
        if isinstance(mri_input, np.ndarray):
            mri_input = torch.FloatTensor(mri_input).to(self.device)
        
        if len(mri_input.shape) == 3:  # [3, H, W]
            mri_input = mri_input.unsqueeze(0)  # [1, 3, H, W]
        
        # 生成CT图像
        generated_ct = self.generator.sample(mri_input)
        
        return generated_ct
    
    def batch_inference(self, data_loader, save_dir):
        """批量推理"""
        os.makedirs(save_dir, exist_ok=True)
        
        all_metrics = []
        
        for batch_idx, batch in enumerate(data_loader):
            mri_input = batch['mri'].to(self.device)
            ct_target = batch['ct'].to(self.device)
            
            # 生成CT图像
            generated_ct = self.generate_ct(mri_input)
            
            # 计算指标
            metrics = calculate_metrics(generated_ct, ct_target)
            all_metrics.append(metrics)
            # 保存结果图像
            save_sample_images(
                mri_input, ct_target, generated_ct,
                os.path.join(save_dir, f'batch_{batch_idx}.png')
            )
            
            print(f"Batch {batch_idx}: SSIM={metrics['SSIM']:.4f}, "
                  f"PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.4f}")
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print(f"\n平均指标:")
        print(f"SSIM: {avg_metrics['SSIM']:.4f}")
        print(f"PSNR: {avg_metrics['PSNR']:.2f}")
        print(f"MAE: {avg_metrics['MAE']:.4f}")
        
        return avg_metrics
    
    def single_inference(self, mri_path_dict, save_path=None):
        """单个样本推理
        Args:
            mri_path_dict: {'T1W': path, 'T2W': path, 'mask': path}
            save_path: 保存路径
        """
        # 加载MRI数据
        import nibabel as nib
        
        t1w_img = nib.load(mri_path_dict['T1W']).get_fdata()
        t2w_img = nib.load(mri_path_dict['T2W']).get_fdata()
        mask_img = nib.load(mri_path_dict['mask']).get_fdata()
        
        # 选择中间切片
        slice_idx = t1w_img.shape[2] // 2
        t1w_slice = t1w_img[:, :, slice_idx]
        t2w_slice = t2w_img[:, :, slice_idx]
        mask_slice = mask_img[:, :, slice_idx]
        
        # 归一化
        def normalize_slice(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            return img * 2.0 - 1.0
        
        t1w_slice = normalize_slice(t1w_slice)
        t2w_slice = normalize_slice(t2w_slice)
        mask_slice = normalize_slice(mask_slice)
        
        # 组合为多模态输入
        mri_input = np.stack([t1w_slice, t2w_slice, mask_slice], axis=0)
        mri_tensor = torch.FloatTensor(mri_input).unsqueeze(0).to(self.device)
        
        # 调整大小到模型输入尺寸
        mri_tensor = torch.nn.functional.interpolate(
            mri_tensor, size=(256, 256), mode='bilinear', align_corners=False
        )
        
        # 生成CT图像
        generated_ct = self.generate_ct(mri_tensor)
        
        # 保存结果
        if save_path:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(mri_tensor[0, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('T1W')
            axes[0].axis('off')
            
            axes[1].imshow(mri_tensor[0, 1].cpu().numpy(), cmap='gray')
            axes[1].set_title('T2W')
            axes[1].axis('off')
            
            axes[2].imshow(mri_tensor[0, 2].cpu().numpy(), cmap='gray')
            axes[2].set_title('Mask')
            axes[2].axis('off')
            
            axes[3].imshow(generated_ct[0, 0].cpu().numpy(), cmap='gray')
            axes[3].set_title('Generated CT')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return generated_ct

def main():
    parser = argparse.ArgumentParser(description='MLDCGAN Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_root', type=str, help='Root directory of test data')
    parser.add_argument('--save_dir', type=str, default='inference_results', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--single_inference', action='store_true', help='Run single inference')
    parser.add_argument('--t1w_path', type=str, help='T1W image path for single inference')
    parser.add_argument('--t2w_path', type=str, help='T2W image path for single inference')
    parser.add_argument('--mask_path', type=str, help='Mask image path for single inference')
    
    args = parser.parse_args()
    
    # 简单配置
    class SimpleConfig:
        def __init__(self):
            self.image_size = 256
            self.channels = 1
            self.latent_dim = 512
            self.num_timesteps = 4
            self.beta_1 = 1e-4
            self.beta_T = 0.02
    
    config = SimpleConfig()
    
    # 创建推理器
    inference = MLDCGANInference(args.model_path, config)
    
    if args.single_inference:
        # 单个推理
        if not all([args.t1w_path, args.t2w_path, args.mask_path]):
            print("单个推理需要提供T1W、T2W和mask路径")
            return
        
        mri_paths = {
            'T1W': args.t1w_path,
            'T2W': args.t2w_path,
            'mask': args.mask_path
        }
        
        save_path = os.path.join(args.save_dir, 'single_result.png')
        os.makedirs(args.save_dir, exist_ok=True)
        
        generated_ct = inference.single_inference(mri_paths, save_path)
        print(f"单个推理完成，结果保存到: {save_path}")
    
    else:
        # 批量推理
        if not args.data_root:
            print("批量推理需要提供数据根目录")
            return
        
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
        ])
        
        test_dataset = MRICTDataset(args.data_root, mode='test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        metrics = inference.batch_inference(test_loader, args.save_dir)
        print("批量推理完成")

if __name__ == '__main__':
    main()

            
