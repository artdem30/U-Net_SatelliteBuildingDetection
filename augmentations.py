import albumentations as albu

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.RandomCrop(height=832, width=832, always_apply=True),
        albu.PadIfNeeded(min_height=800, min_width=800, always_apply=True, border_mode=0),

        albu.GaussNoise(p=0.3),
        albu.Perspective(p=0.3),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1)
            ],
            p=0.3,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
                albu.InvertImg(p=1),
                albu.RGBShift(p=1)
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=1504, min_width=1504, always_apply=True, border_mode=0),
    ]
    return albu.Compose(test_transform)

def get_prediction_augmentation():
    return albu.Compose([albu.PadIfNeeded(min_height=960, min_width=960, always_apply=True, border_mode=0)])

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(transform)