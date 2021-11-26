import random
import torch
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
    
class DataAugmentation(object):
    def __init__(self, p=0.5, augmentation_type=None, min_factor=None, max_factor=None):
        self.p = p
        self.min_factor = min_factor
        self.max_factor = max_factor
        if augmentation_type is None:
            self.func = None
        elif augmentation_type == 'contrast':
            self.func = F.adjust_contrast
        elif augmentation_type == 'brightness':
            self.func = F.adjust_brightness
        elif augmentation_type == 'saturation':
            self.func = F.adjust_saturation
        elif augmentation_type == 'gamma':
            self.func = F.adjust_gamma
        elif augmentation_type == 'hue':
            self.func = F.adjust_hue
        else:
            raise ValueError('NON valid augmentation type !')
            
    def __call__(self, image, target=None):
        if self.func is not None \
        and random.random() > self.p:
            image = self.func(image, random.uniform(self.min_factor, self.max_factor))
            
        if target is not None:
            return image, target
        else:
            return image
    

class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            return image, target
        else:
            return image
