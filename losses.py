# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLDCGANLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_l1=100.0, lambda_fm=10.0, lambda_diffusion=1.0):
        super(MLDCGANLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_fm = lambda_fm
        self.lambda_diffusion = lambda_diffusion
        
        self.adversarial_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def discriminator_loss(self, real_pred, fake_pred):
        """判别器损失"""
        real_loss = 0
        fake_loss = 0
        
        # 处理多尺度判别器输出
        if isinstance(real_pred, list):
            for r_pred, f_pred in zip(real_pred, fake_pred):
                real_target = torch.ones_like(r_pred)
                fake_target = torch.zeros_like(f_pred)
                
                real_loss += self.adversarial_loss(r_pred, real_target)
                fake_loss += self.adversarial_loss(f_pred, fake_target)
        else:
            real_target = torch.ones_like(real_pred)
            fake_target = torch.zeros_like(fake_pred)
            
            real_loss = self.adversarial_loss(real_pred, real_target)
            fake_loss = self.adversarial_loss(fake_pred, fake_target)
        
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss
    
    def generator_adversarial_loss(self, fake_pred):
        """生成器对抗损失"""
        adv_loss = 0
        
        if isinstance(fake_pred, list):
            for f_pred in fake_pred:
                fake_target = torch.ones_like(f_pred)
                adv_loss += self.adversarial_loss(f_pred, fake_target)
        else:
            fake_target = torch.ones_like(fake_pred)
            adv_loss = self.adversarial_loss(fake_pred, fake_target)
        
        return adv_loss * self.lambda_adv
    
    def feature_matching_loss(self, real_features, fake_features):
        """特征匹配损失"""
        fm_loss = 0
        
        if isinstance(real_features[0], list):  # 多尺度
            for r_feats, f_feats in zip(real_features, fake_features):
                for r_feat, f_feat in zip(r_feats, f_feats):
                    fm_loss += self.l1_loss(f_feat, r_feat.detach())
        else:  # 单尺度
            for r_feat, f_feat in zip(real_features, fake_features):
                fm_loss += self.l1_loss(f_feat, r_feat.detach())
        
        return fm_loss * self.lambda_fm
    
    def reconstruction_loss(self, generated, target):
        """重构损失（L1损失）"""
        return self.l1_loss(generated, target) * self.lambda_l1
    
    def diffusion_loss(self, predicted_noise, actual_noise):
        """扩散去噪损失"""
        return self.mse_loss(predicted_noise, actual_noise) * self.lambda_diffusion
    
    def perceptual_loss(self, generated, target, feature_extractor):
        """感知损失（使用预训练特征提取器）"""
        generated_features = feature_extractor(generated)
        target_features = feature_extractor(target)
        
        perceptual_loss = 0
        for g_feat, t_feat in zip(generated_features, target_features):
            perceptual_loss += self.mse_loss(g_feat, t_feat)
        
        return perceptual_loss
