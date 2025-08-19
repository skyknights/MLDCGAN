# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms

class MRICTDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        
        # 假设数据结构为：data_root/train(or test)/patient_id/
        # 包含：T1W.nii, T2W.nii, mask.nii, CT.nii
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        data_list = []
        mode_path = os.path.join(self.data_root, self.mode)
        
        for patient_id in os.listdir(mode_path):
            patient_path = os.path.join(mode_path, patient_id)
            if os.path.isdir(patient_path):
                t1w_path = os.path.join(patient_path, 'T1W.nii')
                t2w_path = os.path.join(patient_path, 'T2W.nii')
                mask_path = os.path.join(patient_path, 'mask.nii')
                ct_path = os.path.join(patient_path, 'CT.nii')
                
                if all(os.path.exists(p) for p in [t1w_path, t2w_path, mask_path, ct_path]):
                    data_list.append({
                        'T1W': t1w_path,
                        'T2W': t2w_path,
                        'mask': mask_path,
                        'CT': ct_path
                    })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_paths = self.data_list[idx]
        
        # 加载NIfTI文件
        t1w_img = nib.load(data_paths['T1W']).get_fdata()
        t2w_img = nib.load(data_paths['T2W']).get_fdata()
        mask_img = nib.load(data_paths['mask']).get_fdata()
        ct_img = nib.load(data_paths['CT']).get_fdata()
        
        # 选择中间切片或随机切片
        slice_idx = t1w_img.shape[2] // 2  # 中间切片
        
        t1w_slice = t1w_img[:, :, slice_idx]
        t2w_slice = t2w_img[:, :, slice_idx]
        mask_slice = mask_img[:, :, slice_idx]
        ct_slice = ct_img[:, :, slice_idx]
        
        # 归一化到[-1, 1]
        t1w_slice = self._normalize(t1w_slice)
        t2w_slice = self._normalize(t2w_slice)
        mask_slice = self._normalize(mask_slice)
        ct_slice = self._normalize(ct_slice)
        
        # 转换为tensor
        t1w_tensor = torch.FloatTensor(t1w_slice).unsqueeze(0)
        t2w_tensor = torch.FloatTensor(t2w_slice).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask_slice).unsqueeze(0)
        ct_tensor = torch.FloatTensor(ct_slice).unsqueeze(0)
        
        # 多模态MRI输入
        mri_input = torch.cat([t1w_tensor, t2w_tensor, mask_tensor], dim=0)
        
        if self.transform:
            mri_input = self.transform(mri_input)
            ct_tensor = self.transform(ct_tensor)
        
        return {
            'mri': mri_input,  # [3, H, W] - T1W, T2W, mask
            'ct': ct_tensor    # [1, H, W] - target CT
        }
    
    def _normalize(self, img):
        """归一化图像到[-1, 1]"""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img = img * 2.0 - 1.0
        return img

def get_data_loaders(config):
    """获取训练和测试数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
    ])
    
    train_dataset = MRICTDataset(
        config.data_root, 
        mode='train', 
        transform=transform
    )
    
    test_dataset = MRICTDataset(
        config.data_root, 
        mode='test', 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    return train_loader, test_loader
