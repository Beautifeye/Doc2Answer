import os
import shutil
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import cv2
import numpy as np
import albumentations as A
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from faster_RCNN.stamp_dataset_class import StampDataset
from faster_RCNN import utils
from faster_RCNN import transforms as T
from download_models import download_model


####################################################################################
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        self.feature_extractor.classifier[1] = nn.Conv2d(512, 2048, kernel_size=(1,1), stride=(1,1))
        self.linear = nn.Sequential(nn.Linear(2048, 2048), nn.Sigmoid())

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = (self.linear(x) * 2.) - 1
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
        return x

    def forward(self, x_pair):
        x1, x2 = x_pair[:,:3], x_pair[:,3:]
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        sim = (torch.sum(torch.mul(out1, out2), dim=1) + 1) / 2.
        return sim
    
    
def preprocesses(img, boxes, show=False):
    patches = None
    img_patches = []
    dummy_transforms = A.Compose([A.Resize(128, 128)])
    for box in boxes:
        patch = img[box[1]:box[3], box[0]:box[2]].copy().astype(np.float32) / 255.
        patch = dummy_transforms(image=patch)['image']
        if show:
            img_patches.append(patch.copy())
        patch = ToTensor()(patch)[None, :]
        if patches is None:
            patches = patch
        else:
            patches = torch.cat([patches, patch], dim=0)
    if show:
        return patches, img_patches
    return patches
####################################################################################

    
parser = argparse.ArgumentParser(description='Stamps detection and classification for SocialCard documents')
parser.add_argument('-i','--input_img',  help='path to the doc to parse', required=True)
parser.add_argument('-o','--output_dir', help='path to the directory to save all outputs to', required=True)
args = parser.parse_args()

# create empty dir to store output files into
if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

# load stamps detector model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
download_model("stamp-detector.pt")
checkpoint_path = os.path.join("models", "stamp-detector.pt")
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
model.to(device)
model.eval()
print('[INFO] Stamps detector model loaded successfully')

# load stamps siamese network
siamese_model = Siamese()
download_model("stamp-classifier.pt")
checkpoint_path = os.path.join("models", "stamp-classifier.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
siamese_model.load_state_dict(checkpoint)
siamese_model.to(device)
siamese_model.eval()
print('[INFO] Stamps siamese model loaded successfully')

# create dataset object for the input image
ds = StampDataset(
    data_df=pd.DataFrame({'path':[args.input_img], 'points':[[]]}),
    transforms=T.Compose([T.ToTensor()])
)

# run stamps detection
with torch.no_grad():
    preds = [
        {k: v.cpu().detach().numpy() for k,v in res.items()}
        for res in model(ds[0][0].to(device)[None])
    ]
boxes = [preds[0]['boxes'][i] for i in range(len(preds[0]['boxes'])) if preds[0]['scores'][i] > 0.5]
boxes = np.asarray(boxes).astype(int)

# create output image with all detected stamps
img = cv2.imread(args.input_img)
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2)
cv2.imwrite(os.path.join(args.output_dir, "stamps-detection.jpg"), img)
print('[INFO] Stamps detection done')

# extract detected stamps from the input image and compute embieddings through the siamese network
img = cv2.cvtColor(cv2.imread(args.input_img), cv2.COLOR_BGR2RGB)
patches = preprocesses(img, boxes)
with torch.no_grad():
    vecs = siamese_model.forward_one(patches.to(device)).detach().cpu().numpy()
    
# match the computed embeddings against the template ones
template_vecs = pkl.load(open('./stamp_classification/template_vecs.pkl','rb+'))
classes = []
for i in range(len(vecs)):
    scores_i = np.sum(vecs[i]* template_vecs, axis=1).ravel()
    class_i = np.argmax(scores_i)
    classes.append(class_i)
    
    
# create output image with all detected stamps
class2color = {c: tuple(int(np.random.randint(0,256)) for _ in range(3)) for c in np.unique(classes)}
img = cv2.imread(args.input_img)
for (x1, y1, x2, y2), c in zip(boxes, classes):
    cv2.rectangle(img, (x1, y1), (x2, y2), class2color[c], 2)
cv2.imwrite(os.path.join(args.output_dir, "stamps-detection-classification.jpg"), img)
print('[INFO] Stamps classification done')

# create csv including all predictions
pd.DataFrame({"box": [box for box in boxes], "classes": classes}).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
print('[INFO] Done')