import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np


class StampDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transforms=None):
        self.data_df = data_df.copy()
        self.card_paths = sorted(self.data_df.path.unique())
        self.transforms = transforms
        

    def __getitem__(self, idx):
        
        img_path = self.card_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        boxes = self.data_df.loc[self.data_df.path==img_path, 'points'].apply(np.array).values
        boxes = np.stack(boxes)
        num_objs = len(boxes)
#         xmin, ymin, xmax, ymax = item['points']
#         boxes = [[xmin, ymin, xmax, ymax]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.view(-1,4)
        
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {'boxes': boxes,
                  'labels': labels,
                  'image_id': image_id,
                  'area': area,
                  'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.data_df.path.nunique()