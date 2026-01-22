import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, subset_path, transform=None):
        self.subset_path = subset_path
        self.transform = transform
        
        # 이미지와 미리 생성된 PNG 마스크 경로 설정
        self.img_path = os.path.join(self.subset_path, 'images')
        self.mask_png_path = os.path.join(self.subset_path, 'masks_png') # 변경된 부분
        
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"❌ 경로 확인 필요: {self.img_path}")
            
        self.image_names = sorted([f for f in os.listdir(self.img_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_full_path = os.path.join(self.img_path, img_name)
        
        # 1. 이미지 로드
        image = cv2.imread(img_full_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 2. 미리 생성된 PNG 마스크 로드 (JSON 파싱 로직 제거됨)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_full_path = os.path.join(self.mask_png_path, mask_name)
        
        if os.path.exists(mask_full_path):
            # IMREAD_GRAYSCALE로 읽어야 0, 1, 2, 3 정수 값이 유지됨
            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Albumentations 적용
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()
            
        return {'image': image, 'mask': mask}