from torchvision.transforms import v2 as T
import torch


def labels_getter(*inputs):
    target = inputs[0][1]
    result = [target["labels"]]
    if "scores" in target.keys():
        result.append(target["scores"])
    return tuple(result)


def get_transform_supervised():
    transforms = [
                  T.RandomIoUCrop(0.7, 1, 0.5, 2),
                  T.ColorJitter(0.4, 0.6,  0.6, 0.05),
                  T.RandomHorizontalFlip(p=0.5),
                  T.ClampBoundingBoxes(),
                  T.SanitizeBoundingBoxes(labels_getter=labels_getter),
                  T.ToDtype(torch.float, scale=True)
                  ]
    return T.Compose(transforms)


def get_transform_unsupervised_weak():
    transforms = [
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.6, 0.6, 0.05),
        T.ToDtype(torch.float, scale=True)
    ]
    return T.Compose(transforms)


def get_transform_unsupervised_strong():
    transforms = [
                  T.RandomIoUCrop(0.7, 1, 0.5, 2),
                  T.RandomPhotometricDistort(),
                  T.RandomHorizontalFlip(p=0.5),
                  T.RandomErasing(0.5, (0.01, 0.1), (0.3, 3.3)),
                  T.ClampBoundingBoxes(),
                  T.SanitizeBoundingBoxes(labels_getter=labels_getter),
                  T.ToDtype(torch.float, scale=True)
                  ]
    return T.Compose(transforms)


def get_transform_valid():
    transforms = [
                  T.ToDtype(torch.float, scale=True)
                  ]
    return T.Compose(transforms)
