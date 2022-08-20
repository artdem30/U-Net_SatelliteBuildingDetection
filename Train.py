import os
import augmentations as aug
from load_data import BuildingDataset
import torch
from torch.nn import ReLU
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

DATA_DIR = 'dataset//'
train_img_dir = os.path.join(DATA_DIR, 'train', 'images')
train_mask_dir = os.path.join(DATA_DIR, 'train', 'masks')
valid_img_dir = os.path.join(DATA_DIR, 'valid', 'images')
valid_mask_dir = os.path.join(DATA_DIR, 'valid', 'masks')
test_dir = os.path.join(DATA_DIR, 'test')

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = BuildingDataset(
    train_img_dir,
    train_mask_dir,
    augmentation=aug.get_training_augmentation(),
    preprocessing=aug.get_preprocessing(preprocessing_fn)
)

valid_dataset = BuildingDataset(
    valid_img_dir,
    valid_mask_dir,
    augmentation=aug.get_validation_augmentation(),
    preprocessing=aug.get_preprocessing(preprocessing_fn)
)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 60 epochs
max_score = 0

for i in range(60):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 20:
        optimizer.param_groups[0]['lr'] = .00005
        print('Decrease decoder learning rate to 0.00005!')
    elif i == 30:
        optimizer.param_groups[0]['lr'] = .00001
        print('Decrease decoder learning rate to 1e-5!')
