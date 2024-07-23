import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from asset import img_transform, mask_transform

class SSF_SH_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, edge_dilate=1, resize=(416, 416)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        assert os.path.exists(image_dir), f"Not found {image_dir}"
        assert os.path.exists(mask_dir), f"Not found {mask_dir}"

        self.transform_image = img_transform(resize)
        self.transform_mask = mask_transform(resize)

        # preprocessing
        self.transform_to_tensor = transforms.ToTensor()
        self.dilate = edge_dilate
        self.resize = resize
        
        # image list
        self.img_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        img_filename = self.img_filenames[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename.replace('.jpg', '.png'))
        
        original_image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # meta data
        meta = {}
        meta['filename'] = img_filename
        meta['shape'] = original_image.size
        meta['mask_path'] = mask_path
        meta['image_path'] = img_path
        
        # hsv image
        hsv_img = original_image.convert("HSV")
        hsv = transforms.functional.to_tensor(hsv_img)
        hsv = transforms.functional.resize(hsv, self.resize) 
        
        # preprocessing
        input_image = self.transform_image(original_image)
        gt_mask = self.transform_mask(mask)
        gt_mask = torch.where(gt_mask > 0.5, 1.0, 0.0)
        

        # エッジ画像（配列の生成）
        mask_resized = cv2.resize(np.array(mask), self.resize, interpolation=cv2.INTER_NEAREST)
        edge_canny = cv2.Canny(mask_resized * 255, 0, 255)
        edge_canny[edge_canny < 127] = 0
        edge_canny[edge_canny > 127] = 1
        edge = cv2.dilate(edge_canny, np.ones((self.dilate, self.dilate), np.uint8))
        gt_edge = torch.tensor(np.expand_dims(edge, 0).astype(np.float32))

        original_image = np.asarray(original_image.resize(self.resize)).copy()
        original_image.flags.writeable = True

        return input_image, hsv[1:], gt_mask, gt_edge, meta