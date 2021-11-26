import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image
import numpy as np


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transforms=None):
        self.data_df = data_df
        self.transforms = transforms
        self.imgs = self.data_df.path.unique()
        

    def __getitem__(self, idx):
        
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        
        boxes = np.array([np.array(points) for points in self.data_df.query("path == @img_path")['points'].values])
        num_objs = len(boxes)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.view(-1,4)
        
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    

    def __len__(self):
        return len(self.imgs)