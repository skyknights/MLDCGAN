# generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from multimodal_fusion import MultimodalFusionModule
from latent_diffusion import UNetLatentEncoder, LatentDiffusionProcess, ResBlock, AttentionBlock, PositionalEncoding

class MLDCGANGenerator(nn.Module):
    def __init__(self, config):
        super(MLDCGANGenerator, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_timesteps = config.num_timesteps
        
        # 多模态融合模块
        self.multimodal_fusion = MultimodalFusionModule(
            in_channels=3,  # T1W, T2W, mask
            out_channels=64
        )
        
        # 潜在扩散过程
        self.diffusion_process = LatentDiffusionProcess(
            num_timesteps=config.num_timesteps,
            beta_1=config.beta_1,
            beta_T=config.beta_T
        )
        
        # 潜在编码器（用于训练时的噪声预测）
        self.latent_encoder = UNetLatentEncoder(
            in_channels=1,  # CT图像
            latent_dim=config.latent_dim,
            time_emb_dim=128
        )
        
        # 时间嵌入
        self.time_embedding = nn.Sequential(
            PositionalEncoding(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )
        
        # 条件嵌入（多模态特征）
        self.condition_embedding = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        
        # 主干生成网络 - UNet结构
        self.encoder1 = self._make_encoder_block(self.latent_dim + 128, 64)  # latent + condition
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResBlock(512, 512, 128),
            AttentionBlock(512),
            ResBlock(512, 512, 128)
        )
        
        # Decoder
        self.decoder4 = self._make_decoder_block(512 + 512, 256)  # skip connection
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        self.decoder2 = self._make_decoder_block(128 + 128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 32)
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        # 潜在空间到特征映射
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * 64),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (1, 64, 64))
        )
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, multimodal_mri, timestep=None, latent_code=None, mode='generate'):
        """
        Args:
            multimodal_mri: [B, 3, H, W] - T1W, T2W, mask
            timestep: [B] - diffusion timestep
            latent_code: [B, latent_dim] - latent code for generation
            mode: 'generate' or 'denoise'
        """
        batch_size = multimodal_mri.size(0)
        
        # 1. 多模态特征提取和融合
        multimodal_features = self.multimodal_fusion(multimodal_mri)  # [B, 64, H, W]
        condition_emb = self.condition_embedding(multimodal_features)  # [B, 128, H, W]
        
        if mode == 'generate':
            # 生成模式：从随机噪声或给定的潜在编码生成
            if latent_code is None:
                latent_code = torch.randn(batch_size, self.latent_dim, device=multimodal_mri.device)
            
            # 如果有时间步，进行扩散去噪
            if timestep is not None:
                time_emb = self.time_embedding(timestep)  # [B, 128]
                # 在潜在空间中添加噪声
                noisy_latent, _ = self.diffusion_process.add_noise(latent_code.unsqueeze(-1).unsqueeze(-1), timestep)
                latent_code = noisy_latent.squeeze(-1).squeeze(-1)
            
            # 将潜在编码转换为特征图
            latent_features = self.latent_to_features(latent_code)  # [B, 1, 64, 64]
            latent_features = F.interpolate(latent_features, size=multimodal_mri.shape[2:], mode='bilinear', align_corners=False)
            
            # 组合潜在特征和条件特征
            combined_features = torch.cat([latent_features, condition_emb], dim=1)  # [B, latent_dim+128, H, W]
            
        elif mode == 'denoise':
            # 去噪模式：预测噪声
            # 这个模式主要用于训练扩散过程
            if timestep is None:
                timestep = self.diffusion_process.sample_timestep(batch_size, multimodal_mri.device)
            
            time_emb = self.time_embedding(timestep)
            # 在这种情况下，latent_code实际上是噪声CT图像
            combined_features = torch.cat([latent_code, condition_emb], dim=1)
        
        # 2. UNet编码器
        enc1 = self.encoder1(combined_features)  # [B, 64, H/2, W/2]
        enc2 = self.encoder2(enc1)              
        enc1 = self.encoder1(combined_features)  # [B, 64, H/2, W/2]
        enc2 = self.encoder2(enc1)              # [B, 128, H/4, W/4]
        enc3 = self.encoder3(enc2)              # [B, 256, H/8, W/8]
        enc4 = self.encoder4(enc3)              # [B, 512, H/16, W/16]
        
        # 3. Bottleneck处理
        bottleneck = self.bottleneck(enc4)      # [B, 512, H/16, W/16]
        
        # 4. UNet解码器（带跳跃连接）
        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))  # [B, 256, H/8, W/8]
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))        # [B, 128, H/4, W/4]
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))        # [B, 64, H/2, W/2]
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))        # [B, 32, H, W]
        
        # 5. 输出层
        output = self.output_conv(dec1)  # [B, 1, H, W]
        
        return output
    
    def sample(self, multimodal_mri, num_steps=None):
        """采样生成CT图像"""
        if num_steps is None:
            num_steps = self.num_timesteps
            
        batch_size = multimodal_mri.size(0)
        device = multimodal_mri.device
        
        # 从纯噪声开始
        x = torch.randn(batch_size, self.latent_dim, device=device)
        
        # 逐步去噪
        for t in reversed(range(num_steps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # 预测噪声
                predicted_noise = self.forward(multimodal_mri, timestep, x, mode='denoise')
                
                # 去噪步骤
                alpha = self.diffusion_process.alphas[t]
                alpha_cumprod = self.diffusion_process.alphas_cumprod[t]
                beta = self.diffusion_process.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
        
        # 最终生成
        generated_ct = self.forward(multimodal_mri, latent_code=x, mode='generate')
        return generated_ct

