import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
from scipy.spatial import distance_matrix
import cv2


class DigitsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, X_mnist, Y_mnist, transforms=None):
        self.X = X
        self.Y = Y
        self.X_mnist = X_mnist
        self.Y_mnist = Y_mnist
        self.transforms = transforms     

    def __getitem__(self, idx):
        
        img = np.zeros((28*3, 28*7), dtype=np.uint8)
        boxes = []
        n_digits = np.random.randint(0, 7)
        ids = np.random.randint(0, len(self.X), size=(n_digits))
        shapes = None
        while shapes is None or shapes.sum() > (28*7):
            shapes = np.random.randint(21, 28*3, size=(n_digits))
        labels = self.Y[ids].ravel()
        labels = np.concatenate([labels, [10]*(6-len(labels))])
        done = False

        img_end_x, img_end_y = 0, 0
        
        if n_digits > 0:
            y_avg =  np.random.randint(-shapes[0]//5, 28*3-int(4*shapes[0]//5))
            
        for i in range(n_digits):
            if np.random.random() < .5:
                X_i = self.X[ids[i]].copy().reshape(28, 28).T
            else:
                idx = np.random.randint(len(self.X_mnist))
                X_i = self.X_mnist[idx]
                labels[i] = self.Y_mnist[idx]

            X_i = cv2.resize(X_i, (shapes[i], shapes[i]))
            start_x = np.random.randint(img_end_x-10, 28*7-shapes[i:].sum()+7)
            start_y = np.clip(y_avg + np.random.randint(-shapes[i]//5, shapes[i]//5), -shapes[i]//5, 28*3-int(4*shapes[i]//5))
            img_start_x, img_start_y = max(0, start_x), max(0, start_y)
            img_end_x, img_end_y = min(start_x+shapes[i], img.shape[1]), min(start_y+shapes[i], img.shape[0])
            digit_start_x, digit_start_y = -start_x if start_x < 0 else 0, -start_y if start_y < 0 else 0
            digit_end_x, digit_end_y = min(shapes[i], int(img.shape[1]) - start_x), min(shapes[i], int(img.shape[0]) - start_y)
            
            block = img[img_start_y:img_end_y, img_start_x:img_end_x].copy()
            
            cropped_digit = X_i[digit_start_y:digit_end_y ,digit_start_x:digit_end_x].copy()

            img[img_start_y:img_end_y, img_start_x:img_end_x] = np.clip(block + cropped_digit, 0, 255)
            
            box = [start_x, start_y, start_x+shapes[i], start_y+shapes[i]]
            boxes.append(box)
        if len(boxes) > 0:
            boxes = np.stack(boxes)
        else:
            boxes = np.zeros((0,4))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.view(-1, 4)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((n_digits,), dtype=torch.int64)
        
#         img = Image.fromarray(img).convert('RGB')

        target = {'boxes': boxes,
                  'labels': labels,
                  'image_id': image_id,
                  'area': area,
                  'iscrowd': iscrowd}
        
        img = img.astype(np.float32)/255.
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.
        img = img.astype(np.uint8)
        
        if self.transforms is not None:
            img = torch.Tensor(self.transforms(image=img)['image']) / 255.
        img = torch.cat([img[None]]*3, dim=0)
        return img, labels
    
    def preprocess(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 255 - img
        
        img = img.astype(np.float32)/255.
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.
        img = img.astype(np.uint8)
        
        if self.transforms is not None:
            img = torch.Tensor(self.transforms(image=img)['image']) / 255.
        img = torch.cat([img[None]]*3, dim=0)
        return img

    def __len__(self):
        return len(self.X) // 6
    
    
    
class CellsDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_folder, transforms=None, mode='test'):
        self.df = df
        self.img_folder = img_folder
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_folder, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = row['points']
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if self.mode == 'train':
            x1 += (np.random.random()*.2-.1) * dx
            x2 += (np.random.random()*.2-.1) * dx
            y1 += (np.random.random()*.2-.1) * dy
            y2 += (np.random.random()*.2-.1) * dy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = img[y1:y2, x1:x2]
        if np.prod(img.shape)==0:
            print(idx, img_name, (x1, y1, x2, y2))
            return self.__getitem__(np.random.randint(len(self)))
        value = ''.join([v for v in row['value'].replace(' ','').replace('.','-').replace('b','') if v.isalnum() or v=='-'])
        an_value = [v for v in value if v.isalnum()]
        if value=='none':
            labels = [10]*6
        elif len(an_value)<4:
            try:
                labels = [int(v) for v in an_value]+[10]*(6-len(an_value))
            except:
                print(value, an_value, img_name)
                raise
        elif len(an_value)==len(value)==4:
            try:
                labels = [int(v) for v in an_value]+[10]*(6-len(an_value))
            except:
                print(value, an_value, img_name)
                raise
        else:
            try:
                day = value.split('-')[0].zfill(2)
                month = value.split('-')[1].zfill(2)
                year = value.split('-')[-1][-2:].zfill(2)
                labels = [int(d) for split in [day, month, year] for d in split]
            except:
                print(value, img_name)
                raise
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transforms is not None:
            img = torch.Tensor(self.transforms(image=img)['image']) / 255.
        img = torch.cat([img[None]]*3, dim=0)
        return img, labels
    
    def preprocess(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 255 - img
        
#         img = img.astype(np.float32)/255.
#         img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.
#         img = img.astype(np.uint8)
        
        if self.transforms is not None:
            img = torch.Tensor(self.transforms(image=img)['image']) / 255.
        img = torch.cat([img[None]]*3, dim=0)
        return img

    def __len__(self):
        return len(self.df)