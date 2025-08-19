# multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(in_channels)
        
    def forward(self, x, context=None):
        if context is None:
            context = x
            
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        context_flat = context.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # Multi-head attention
        Q = self.query(x_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(context_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(context_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, -1, C)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.layer_norm(output + x_flat)  # Residual connection
        
        return output.permute(0, 2, 1).view(B, C, H, W)

class MultimodalFusionModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_layers=3):
        super(MultimodalFusionModule, self).__init__()
        
        # Individual encoders for each modality
        self.t1w_encoder = nn.Sequential(
            nn.Conv2d(1, out_channels//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.t2w_encoder = nn.Sequential(
            nn.Conv2d(1, out_channels//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, out_channels//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(out_channels) for _ in range(num_layers)
        ])
        
        # Fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        
    def forward(self, multimodal_input):
        # Split multimodal input
        t1w = multimodal_input[:, 0:1, :, :]
        t2w = multimodal_input[:, 1:2, :, :]
        mask = multimodal_input[:, 2:3, :, :]
        
        # Individual encoding
        t1w_feat = self.t1w_encoder(t1w)
        t2w_feat = self.t2w_encoder(t2w)
        mask_feat = self.mask_encoder(mask)
        
        # Cross-modal attention
        for attention_layer in self.cross_attention_layers:
            t1w_feat_new = attention_layer(t1w_feat, torch.cat([t2w_feat, mask_feat], dim=1))
            t2w_feat_new = attention_layer(t2w_feat, torch.cat([t1w_feat, mask_feat], dim=1))
            mask_feat_new = attention_layer(mask_feat, torch.cat([t1w_feat, t2w_feat], dim=1))
            
            t1w_feat, t2w_feat, mask_feat = t1w_feat_new, t2w_feat_new, mask_feat_new
        
        # Fusion
        fused_features = torch.cat([t1w_feat, t2w_feat, mask_feat], dim=1)
        output = self.fusion_conv(fused_features)
        
        return output
