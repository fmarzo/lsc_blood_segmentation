import torch
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# for Hemoset
def create_eval_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        v2.ToPureTensor(),
    ])

# for Hemoset
def create_train_transform():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply(
            [
                v2.RandomRotation(
                    degrees=(-10, 10),
                    interpolation=v2.InterpolationMode.NEAREST,
                    fill=0,
                )
            ],
            p=0.5,
        ),

        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.0,
                )
            ],
            p=0.5,
        ),
        
        v2.ToDtype(torch.float32, scale=True),

        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),

        
    ])

# for Bleed_seg
def create_bleed_eval_transform():
    return v2.Compose([
        v2.Resize(
            size=(640, 480), #resize of the images of bleed_seg at the same size of the images of Hemoset
            antialias=True,
        ),

        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),
    ])

# for Bleed_seg
def create_bleed_train_transform():
    return v2.Compose([
        v2.Resize(
            size=(640, 480),
            antialias=True,
        ),

        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply(
            [
                v2.RandomRotation(
                    degrees=(-10, 10),
                    interpolation=v2.InterpolationMode.NEAREST,
                    fill=0,
                )
            ],
            p=0.5,
        ),

        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.0,
                )
            ],
            p=0.5,
        ),

        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),
    ])