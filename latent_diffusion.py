# latent_diffusion.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        
        attention = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(C // self.num_heads), -1)
        h = (attention @ v).view(B, C, H, W)
        return x + self.out(h)

class UNetLatentEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, time_emb_dim=128):
        super().__init__()
        
        self.time_embedding = nn.Sequential(
            PositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64, 64, time_emb_dim),
            ResBlock(64, 64, time_emb_dim)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            ResBlock(128, 128, time_emb_dim),
            ResBlock(128, 128, time_emb_dim)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            ResBlock(256, 256, time_emb_dim),
            ResBlock(256, 256, time_emb_dim),
            AttentionBlock(256)
        )
        
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            ResBlock(512, 512, time_emb_dim),
            ResBlock(512, 512, time_emb_dim),
            AttentionBlock(512)
        )
        
        # Bottleneck to latent space
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim)
        )
        
    def forward(self, x, timestep):
        time_emb = self.time_embedding(timestep)
        
        h1 = self.down1(x)
        h1 = h1 + time_emb.view(-1, 64, 1, 1) if hasattr(h1, 'shape') else h1
        
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        
        latent = self.to_latent(h4)
        return latent

class LatentDiffusionProcess(nn.Module):
    def __init__(self, num_timesteps=4, beta_1=1e-4, beta_T=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Non-parametric noise schedule
        self.beta_1 = beta_1
        self.beta_T = beta_T
        
        # Complex denoising distribution parameters
        self.register_buffer('betas', self._get_noise_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _get_noise_schedule(self):
        """Non-linear noise schedule for faster sampling"""
        timesteps = torch.arange(self.num_timesteps).float()
        # Non-parametric schedule: exponential decay
        betas = self.beta_1 * torch.exp(timesteps * torch.log(torch.tensor(self.beta_T / self.beta_1)) / (self.num_timesteps - 1))
        return betas
        
    def add_noise(self, x_0, timestep):
        """Add noise according to the diffusion process"""
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timestep])
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, None, None, None]
        
        noisy_x = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return noisy_x, noise
        
    def sample_timestep(self, batch_size, device):
        """Sample random timesteps"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
