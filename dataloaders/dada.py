import os
import numpy as np
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.mapping import APOLLO2CS, APOLLO16
import torch


class DADARGBEvent(data.Dataset):
    """return dict with img, event, label from DADA"""
    # 18 for Apolloscape, 19 for Cityscapes
    NUM_CLASSES = 19

    def __init__(self, args, root=Path.db_root_dir('dadaevent'), split="val"):
        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.event = {}
        self.labels = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.images[split].sort()
        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.images_base))
        else:
            print("Found %d %s RGB images" % (len(self.images[split]), split))

        self.event_base = os.path.join(self.root, 'event', self.split)
        self.event[split] = self.recursive_glob(rootdir=self.event_base, suffix='.npz')
        self.event[split].sort()

        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        self.labels[split] = self.recursive_glob(rootdir=self.annotations_base, suffix='labelTrainIds.png')
        self.labels[split].sort()

        self.ignore_index = 255

    def __len__(self):
        return len(self.labels[self.split])

    def __getitem__(self, index):
        sample = dict()
        lbl_path = self.labels[self.split][index].rstrip()
        sample['label'] = Image.open(lbl_path) if self.NUM_CLASSES == 19 else self.relabel(lbl_path)

        img_path = self.images[self.split][index].rstrip()
        sample['image'] = Image.open(img_path).convert('RGB')

        event_path = self.event[self.split][index].rstrip()
        sample['event'] = self.get_event(event_path)

        # data augment
        if self.split == 'train':
            raise NotImplementedError
        elif self.split == 'val':
            return self.transform_val(sample), lbl_path
        elif self.split == 'test':
            raise NotImplementedError

    def get_event(self, event_path):
        event_volume = np.load(event_path)['data']
        if self.args.event_dim == 2:
            neg_img = np.sum(event_volume[:9, ...], axis=0, keepdims=True)
            pos_img = np.sum(event_volume[9:, ...], axis=0, keepdims=True)
            event_volume = np.concatenate((neg_img, pos_img), axis=0)
        elif self.args.event_dim == 1:
            neg_img = np.sum(event_volume[:9, ...], axis=0, keepdims=True)
            pos_img = np.sum(event_volume[9:, ...], axis=0, keepdims=True)
            event_volume = neg_img + pos_img
        return event_volume

    def relabel(self, label_path):
        """remove the 'train' class=16."""
        _temp = np.array(Image.open(label_path))
        label_mapping = {m[0]: m[1] for m in APOLLO16}
        for k, v in label_mapping.items():
            _temp[_temp == k] = v
        return Image.fromarray(_temp)


    def recursive_glob(self, rootdir='.', suffix=None):
        if isinstance(suffix, str):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames if filename.endswith(suffix)]
        elif isinstance(suffix, list):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for x in suffix for filename in filenames if filename.startswith(x)]

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.CropBlackArea(crop=[162, 0, 1422, 600]), # left, top, right, bottom
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.event_dim = 2

    dada = DADARGBEvent(args, split='val')
    dataloader = DataLoader(dada, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        sample, _ = sample
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            event = sample['event'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='dada')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(311)
            plt.imshow(img_tmp)
            plt.subplot(312)
            plt.imshow(segmap)
            plt.subplot(313)
            event_neg = np.transpose(event[jj], axes=[1, 2, 0]).astype(np.uint8)[...,0]
            event_neg = Image.fromarray(event_neg, mode='L')
            plt.imshow(event_neg)

        if ii == 1:
            break

    plt.show(block=True)

