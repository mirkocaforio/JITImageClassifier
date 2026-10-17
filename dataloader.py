# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os

import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f"{'train' if train else 'val'}{year}.json")
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, is_test, args):
    """Build dataset for training or validation.
    Args:
        is_train (bool): Whether to build the training dataset.
        is_test (bool): Whether to build the test dataset.
        args: Command-line arguments containing dataset configuration.
    Returns:
        dataset: The constructed dataset object.
        nb_classes: The number of classes in the dataset.
    """
    # transform = build_transform(is_train, args)

    if args.data_set == "RSCD":
        root = os.path.join(
            args.data_path, "train" if is_train else "val" if not is_test else "test"
        )
        # dataset = datasets.ImageFolder(root, transform=build_transform(is_train, args.data_path, args))
        dataset = datasets.ImageFolder(root, transform=build_transform(is_train, args))
        nb_classes = len(dataset.classes)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.data_set}")

    return dataset, nb_classes


# ORIGINALE
def build_transform(is_train, args):
    """Build data transformation pipeline for training or validation.
    Args:
        is_train (bool): Whether to build the transformation for training or validation.
        args: Command-line arguments containing transformation configuration.
    Returns:
        transform: The constructed transformation pipeline.
    """
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.img_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
