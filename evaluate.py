# evaluate.py
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

from generator import MLDCGANGenerator
from data_loader import MRICTDataset, get_data_loaders
from utils import calculate_metrics, denormalize_image
from config import get_config

class MLDCGANEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # 加载模型
        self.generator = MLDCGANGenerator(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"评估模型已加载到设备: {self.device}")
    
    @torch.no_grad()
    def comprehensive_evaluation(self, test_loader):
        """全面评估模型性能"""
        all_ssim = []
        all_psnr = []
        all_mae = []
        all_mse = []
        all_ncc = []  # Normalized Cross Correlation
        
        sample_images = {'mri': [], 'target': [], 'generated': []}
        
        print("开始全面评估...")
        
        for batch_idx, batch in enumerate(test_loader):
            mri_input = batch['mri'].to(self.device)
            ct_target = batch['ct'].to(self.device)
            
            # 生成CT图像
            generated_ct = self.generator.sample(mri_input)
            
            # 计算基本指标
            metrics = calculate_metrics(generated_ct, ct_target)
            all_ssim.append(metrics['SSIM'])
            all_psnr.append(metrics['PSNR'])
            all_mae.append(metrics['MAE'])
            
            # 计算额外指标
            gen_np = denormalize_image(generated_ct).cpu().numpy()
            tar_np = denormalize_image(ct_target).cpu().numpy()
            
            batch_mse = []
            batch_ncc = []
            
            for i in range(gen_np.shape[0]):
                gen_img = gen_np[i, 0].flatten()
                tar_img = tar_np[i, 0].flatten()
                
                # MSE
                mse = mean_squared_error(tar_img, gen_img)
                batch_mse.append(mse)
                
                # NCC (Normalized Cross Correlation)
                ncc = np.corrcoef(gen_img, tar_img)[0, 1]
                if not np.isnan(ncc):
                    batch_ncc.append(ncc)
            
            all_mse.extend(batch_mse)
            all_ncc.extend(batch_ncc)
            
            # 保存一些样本用于可视化
            if batch_idx < 5:  # 只保存前5个批次的样本
                sample_images['mri'].extend(mri_input.cpu())
                sample_images['target'].extend(ct_target.cpu())
                sample_images['generated'].extend(generated_ct.cpu())
            
            if batch_idx % 10 == 0:
                print(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
        
        # 计算统计结果
        results = {
            'SSIM': {
                'mean': np.mean(all_ssim),
                'std': np.std(all_ssim),
                'median': np.median(all_ssim),
                'min': np.min(all_ssim),
                'max': np.max(all_ssim)
            },
            'PSNR': {
                'mean': np.mean(all_psnr),
                'std': np.std(all_psnr),
                'median': np.median(all_psnr),
                'min': np.min(all_psnr),
                'max': np.max(all_psnr)
            },
            'MAE': {
                'mean': np.mean(all_mae),
                'std': np.std(all_mae),
                'median': np.median(all_mae),
                'min': np.min(all_mae),
                'max': np.max(all_mae)
            },
            'MSE': {
                'mean': np.mean(all_mse),
                'std': np.std(all_mse),
                'median': np.median(all_mse),
                'min': np.min(all_mse),
                'max': np.max(all_mse)
            },
            'NCC': {
                'mean': np.mean(all_ncc),
                'std': np.std(all_ncc),
                'median': np.median(all_ncc),
                'min': np.min(all_ncc),
                'max': np.max(all_ncc)
            }
        }
        
        return results, sample_images, {
            'ssim': all_ssim,
            'psnr': all_psnr,
            'mae': all_mae,
            'mse': all_mse,
            'ncc': all_ncc
        }
    
    def save_evaluation_report(self, results, sample_images, raw_metrics, save_dir):
        """保存评估报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存数值结果到CSV
        df_results = pd.DataFrame(results).T
        df_results.to_csv(os.path.join(save_dir, 'evaluation_metrics.csv'))
        
        # 2. 打印结果摘要
        print("\n" + "="*50)
        print("评估结果摘要")
        print("="*50)
        for metric, stats in results.items():
            print(f"{metric}:")
            print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()
        
        # 3. 绘制指标分布图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_data = [
            ('SSIM', raw_metrics['ssim']),
            ('PSNR', raw_metrics['psnr']),
            ('MAE', raw_metrics['mae']),
            ('MSE', raw_metrics['mse']),
            ('NCC', raw_metrics['ncc'])
        ]
        
        for i, (name, data) in enumerate(metrics_data):
            if i < len(axes):
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{name} Distribution')
                axes[i].set_xlabel(name)
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(np.mean(data), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(data):.4f}')
                axes[i].legend()
        
        # 删除多余的子图
        if len(metrics_data) < len(axes):
            for i in range(len(metrics_data), len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 绘制样本结果
        num_samples = min(8, len(sample_images['mri']))
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            mri = sample_images['mri'][i]
            target = sample_images['target'][i]
            generated = sample_images['generated'][i]
            
            # T1W
            axes[i, 0].imshow(mri[0].numpy(), cmap='gray')
            axes[i, 0].set_title('T1W')
            axes[i, 0].axis('off')
            
            # T2W
            axes[i, 1].imshow(mri[1].numpy(), cmap='gray')
            axes[i, 1].set_title('T2W')
            axes[i, 1].axis('off')
            
            # Mask
            axes[i, 2].imshow(mri[2].numpy(), cmap='gray')
            axes[i, 2].set_title('Mask')
            axes[i, 2].axis('off')
            
            # Target CT
            axes[i, 3].imshow(target[0].numpy(), cmap='gray')
            axes[i, 3].set_title('Target CT')
            axes[i, 3].axis('off')
            
            # Generated CT
            axes[i, 4].imshow(generated[0].numpy(), cmap='gray')
            axes[i, 4].set_title('Generated CT')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 绘制相关性矩阵
        metrics_df = pd.DataFrame({
            'SSIM': raw_metrics['ssim'],
            'PSNR': raw_metrics['psnr'], 
            'MAE': raw_metrics['mae'],
            'MSE': raw_metrics['mse'],
            'NCC': raw_metrics['ncc']
        })
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = metrics_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"评估报告已保存到: {save_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MLDCGAN Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of test data')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    # 获取配置（使用默认值）
    config = get_config()
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    
    # 获取测试数据加载器
    _, test_loader = get_data_loaders(config)
    
    # 创建评估器
    evaluator = MLDCGANEvaluator(args.model_path, config)
    
    # 执行全面评估
    results, sample_images, raw_metrics = evaluator.comprehensive_evaluation(test_loader)
    
    # 保存评估报告
    evaluator.save_evaluation_report(results, sample_images, raw_metrics, args.save_dir)

if __name__ == '__main__':
    main()
