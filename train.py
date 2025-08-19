# train.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

from config import get_config
from data_loader import get_data_loaders
from generator import MLDCGANGenerator
from discriminator import MultiScaleDiscriminator
from losses import MLDCGANLoss
from utils import save_checkpoint, load_checkpoint, save_sample_images

class MLDCGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 数据加载器
        self.train_loader, self.test_loader = get_data_loaders(config)
        
        # 模型
        self.generator = MLDCGANGenerator(config).to(self.device)
        self.discriminator = MultiScaleDiscriminator(
            input_channels=4,  # 3(MRI) + 1(CT)
            num_scales=3
        ).to(self.device)
        
        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(0.5, 0.999)
        )
        
        # 学习率调度器
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50, gamma=0.5)
        
        # 损失函数
        self.criterion = MLDCGANLoss(
            lambda_adv=config.lambda_adv,
            lambda_l1=config.lambda_l1,
            lambda_fm=config.lambda_fm
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='logs')
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        # 创建保存目录
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('samples', exist_ok=True)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            mri_input = batch['mri'].to(self.device)  # [B, 3, H, W]
            ct_target = batch['ct'].to(self.device)   # [B, 1, H, W]
            batch_size = mri_input.size(0)
            
            # ================== 训练判别器 ==================
            self.optimizer_D.zero_grad()
            
            # 生成假CT图像
            with torch.no_grad():
                fake_ct = self.generator(mri_input, mode='generate')
            
            # 真实和虚假输入
            real_input = torch.cat([mri_input, ct_target], dim=1)  # [B, 4, H, W]
            fake_input = torch.cat([mri_input, fake_ct.detach()], dim=1)
            
            # 判别器预测
            real_pred = self.discriminator(real_input)
            fake_pred = self.discriminator(fake_input)
            
            # 判别器损失
            d_loss = self.criterion.discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            self.optimizer_D.step()
            
            # ================== 训练生成器 ==================
            self.optimizer_G.zero_grad()
            
            # 生成CT图像
            generated_ct = self.generator(mri_input, mode='generate')
            fake_input = torch.cat([mri_input, generated_ct], dim=1)
            
            # 判别器预测和特征
            fake_pred, fake_features = self.discriminator(fake_input, return_features=True)
            real_pred, real_features = self.discriminator(real_input, return_features=True)
            
            # 生成器损失
            g_adv_loss = self.criterion.generator_adversarial_loss(fake_pred)
            g_l1_loss = self.criterion.reconstruction_loss(generated_ct, ct_target)
            g_fm_loss = self.criterion.feature_matching_loss(real_features, fake_features)
            
            # 扩散损失（训练扩散过程）
            timestep = self.generator.diffusion_process.sample_timestep(batch_size, self.device)
            noisy_ct, noise = self.generator.diffusion_process.add_noise(ct_target, timestep)
            predicted_noise = self.generator.forward(mri_input, timestep, noisy_ct, mode='denoise')
            g_diffusion_loss = self.criterion.diffusion_loss(predicted_noise, noise)
            
            g_total_loss = g_adv_loss + g_l1_loss + g_fm_loss + g_diffusion_loss
            g_total_loss.backward()
            self.optimizer_G.step()
            
            # 统计
            epoch_g_loss += g_total_loss.item()
            epoch_d_loss += d_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'G_loss': f'{g_total_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
            
            # TensorBoard记录
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Loss/Generator', g_total_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/G_Adversarial', g_adv_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/G_L1', g_l1_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/G_FeatureMatch', g_fm_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/G_Diffusion', g_diffusion_loss.item(), self.global_step)
            
            self.global_step += 1
        
        # 学习率调度
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return epoch_g_loss / len(self.train_loader), epoch_d_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.generator.eval()
        
        total_l1_loss = 0
        
        for batch in self.test_loader:
            mri_input = batch['mri'].to(self.device)
            ct_target = batch['ct'].to(self.device)
            
            # 生成CT图像
            generated_ct = self.generator.sample(mri_input)
            
            # 计算L1损失
            l1_loss = self.criterion.l1_loss(generated_ct, ct_target)
            total_l1_loss += l1_loss.item()
        
        avg_l1_loss = total_l1_loss / len(self.test_loader)
        return avg_l1_loss
    
    def train(self):
        """主训练循环"""
        print(f"开始训练MLDCGAN，设备: {self.device}")
        print(f"训练数据: {len(self.train_loader)} batches")
        print(f"测试数据: {len(self.test_loader)} batches")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_g_loss, train_d_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            print(f'Epoch [{epoch+1}/{self.config.num_epochs}]')
            print(f'Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f}')
            print(f'Val L1 Loss: {val_loss:.4f}')
            
            # TensorBoard记录
            self.writer.add_scalar('Epoch/Train_G_Loss', train_g_loss, epoch)
            self.writer.add_scalar('Epoch/Train_D_Loss', train_d_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
            
            # 定期保存检查点
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # 生成样本图像
            if (epoch + 1) % 10 == 0:
                self.generate_samples(epoch + 1)
        
        self.writer.close()
        print("训练完成!")
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, os.path.join('checkpoints', filename))
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join('checkpoints', filename), map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        print(f'已加载检查点: {filename}, Epoch: {self.current_epoch}')
    
    @torch.no_grad()
    def generate_samples(self, epoch):
        """生成样本图像"""
        self.generator.eval()
        
        # 获取一个测试批次
        test_batch = next(iter(self.test_loader))
        mri_input = test_batch['mri'][:4].to(self.device)  # 只取前4个
        ct_target = test_batch['ct'][:4].to(self.device)
        
        # 生成CT图像
        generated_ct = self.generator.sample(mri_input)
        
        # 保存图像
        save_sample_images(
            mri_input, ct_target, generated_ct,
            f'samples/epoch_{epoch}.png'
        )

def main():
    # 获取配置
    config = get_config()
    
    # 创建训练器
    trainer = MLDCGANTrainer(config)
    
    # 如果有检查点，加载它
    if os.path.exists('checkpoints/best_model.pth'):
        try:
            trainer.load_checkpoint('best_model.pth')
        except:
            print("无法加载检查点，从头开始训练")
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
