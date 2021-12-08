import os
import shutil
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import albumentations as A
import torchvision.transforms as T
from download_models import download_model


####################################################################################
class CellDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transforms = T.Compose([T.ToTensor()])
        
    def __getitem__(self, idx):
        x = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transforms(x)
    
    def __len__(self):
        return len(self.img_paths)
    

class HandDigitModel(nn.Module):
    
    def __init__(self):
        super(HandDigitModel, self).__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)        
        self.linear = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 11*6))

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.linear(x)
        return out.view(len(x), 6, 11)
####################################################################################


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MIN_CONFIDENCE = 0.6

parser = argparse.ArgumentParser(description='Cells detection and text extraction for O7 documents')
parser.add_argument('-i','--input_img',  help='path to the doc to parse', required=True)
parser.add_argument('-o','--output_dir', help='path to the directory to save all outputs to', required=True)
args = parser.parse_args()

if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

download_model("text-detector.pt")
model = torch.load("models/text-detector.pt")
model.eval()
model.to(DEVICE)
print("[INFO] Text detector model loaded successfully")

ds = CellDataset([args.input_img])
dl = torch.utils.data.DataLoader(ds, 1)

raw_preds = []
with torch.no_grad():
    for batch in dl:
        raw_preds += model(batch.to(DEVICE))
preds = raw_preds[0]
        
img = cv2.cvtColor(cv2.imread(args.input_img), cv2.COLOR_BGR2RGB)
scores = preds["scores"].cpu().numpy()
above_threshold_ids = scores > MIN_CONFIDENCE
scores = scores[above_threshold_ids]
boxes = preds["boxes"].cpu().numpy()[above_threshold_ids]

colors = [tuple([int(np.random.rand() * 255) for _ in range(3)]) for _ in boxes]
for color, score, box in zip(colors, scores, boxes):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
cv2.imwrite(os.path.join(args.output_dir, "text-detection.jpg"), img)
print("[INFO] Text detection done")

test_transforms = A.Compose([A.Resize(128, 128)])
model = HandDigitModel()
download_model("text-classifier.pt")

model.load_state_dict(torch.load('models/text-classifier.pt', map_location=DEVICE))
model.eval()
model.to(DEVICE)
print("[INFO] Text extractor model loaded successfully")

img = cv2.cvtColor(cv2.imread(args.input_img), cv2.COLOR_RGB2GRAY)
img_bgr = cv2.imread(args.input_img)
all_preds_data = {k:[] for k in ["box", "text"]}

for color, box in zip(colors, boxes):
    x1, y1, x2, y2 = box.astype(int)
    patch = img[y1:y2, x1:x2]
    X_i = test_transforms(image=patch)['image']
    X_i = T.ToTensor()(X_i)
    X_i = torch.cat([X_i]*3, dim=0)[None]

    with torch.no_grad():
        preds = model(X_i.to(DEVICE)).detach().cpu().numpy()
        
    pred_digits = np.argmax(preds[0], axis=1)
    string_result = ''.join([str(el) for i, el in enumerate(pred_digits) if 10 not in pred_digits[:i+1]])
    if len(string_result)==6:
        string_result = string_result[:2]+'-'+string_result[2:4]+'-'+string_result[4:]
    
    all_preds_data["box"].append([x1, y1, x2, y2])
    all_preds_data["text"].append(string_result)
    
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img_bgr, string_result if string_result != "" else "X", (x1, y1 + (y2 - y1) // 2), cv2.FONT_HERSHEY_PLAIN, 3, color)
print("[INFO] Text extraction done")

cv2.imwrite(os.path.join(args.output_dir, "text-detextion-extraction.jpg"), img_bgr)
pd.DataFrame(all_preds_data).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
print("[INFO] Done")