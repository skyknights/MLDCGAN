# discriminator.py
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=4, ndf=64, n_layers=3):  # 4 = 3(MRI) + 1(CT)
        super(PatchGANDiscriminator, self).__init__()
        
        # 初始层
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 中间层
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
        # 特征提取器（用于特征匹配损失）
        self.feature_extractor = nn.ModuleList()
        for i in range(len(sequence) - 1):  # 除了最后一层
            self.feature_extractor.append(sequence[i])
    
    def forward(self, input_tensor, return_features=False):
        """
        Args:
            input_tensor: [B, 4, H, W] - 连接的MRI和CT
            return_features: 是否返回中间特征用于特征匹配
        """
        if return_features:
            features = []
            x = input_tensor
            for layer in self.feature_extractor:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    features.append(x)
            # 最后一层
            x = self.model[-1](x)
            return x, features
        else:
            return self.model(input_tensor)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels=4, ndf=64, n_layers=3, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        
        # 创建多个不同尺度的判别器
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                PatchGANDiscriminator(input_channels, ndf, n_layers)
            )
        
        # 下采样层
        self.downsample = nn.AvgPool2d(2, 2)
    
    def forward(self, input_tensor, return_features=False):
        results = []
        features_list = []
        
        x = input_tensor
        for i, discriminator in enumerate(self.discriminators):
            if return_features:
                result, features = discriminator(x, return_features=True)
                results.append(result)
                features_list.append(features)
            else:
                result = discriminator(x)
                results.append(result)
            
            # 为下一个尺度下采样
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        if return_features:
            return results, features_list
        else:
            return results
