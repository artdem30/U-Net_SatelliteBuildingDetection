import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

DATA_DIR = 'dataset//'

train_img_dir = os.path.join(DATA_DIR, 'train', 'images')
train_mask_dir = os.path.join(DATA_DIR, 'train', 'masks')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class BuildingDataset(Dataset):
    """Building Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None, ):
        self.ids = os.listdir(images_dir)
        self.images_paths = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_paths = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_paths[i])
        mask = cv2.imread(self.masks_paths[i],0)
        mask = np.expand_dims(mask, axis=2)/255

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
