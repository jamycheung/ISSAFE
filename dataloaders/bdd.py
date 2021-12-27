"""
Read BDD10k seg_event
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class BDD(data.Dataset):
    """return dict with img, label of BDD"""
    NUM_CLASSES = 19

    def __init__(self, args, root=Path.db_root_dir('bdd'), split="train"):
        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.event = {}
        self.labels = {}

        with open('dataloaders/bdd_txt/images_{}.txt'.format(split), 'r') as colors_f, \
                open('dataloaders/bdd_txt/events_{}.txt'.format(split), 'r') as events_f, \
                open('dataloaders/bdd_txt/labels_{}.txt'.format(split), 'r') as labels_f:  
            self.images[split] = colors_f.read().splitlines()
            self.event[split] = events_f.read().splitlines()
            self.labels[split] = labels_f.read().splitlines()
            print("Found %d %s RGB images" % (len(self.images[split]), split))
            self.images[split].sort()
            self.event[split].sort()
            self.labels[split].sort()

        self.ignore_index = 255

    def __len__(self):
        return len(self.labels[self.split])

    def __getitem__(self, index):
        sample = dict()
        lbl_path = self.root + self.labels[self.split][index].rstrip()
        sample['label'] = Image.open(lbl_path)

        img_path = self.root + self.images[self.split][index].rstrip()
        sample['image'] = Image.open(img_path).convert('RGB')

        if self.args.event_dim:
            event_path = self.root + self.event[self.split][index].rstrip()
            sample['event'] = self.get_event(event_path)

        # data augment
        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample), lbl_path
        elif self.split == 'test':
            raise NotImplementedError

    def get_event(self, event_path):
        event_volume = np.load(event_path)['data']
        neg_volume = event_volume[:9, ...]
        pos_volume = event_volume[9:, ...]
        if self.args.event_dim == 18:
            event_volume = np.concatenate((neg_volume, pos_volume), axis=0)
        elif self.args.event_dim == 2:
            neg_img = np.sum(neg_volume, axis=0, keepdims=True)
            pos_img = np.sum(pos_volume, axis=0, keepdims=True)
            event_volume = np.concatenate((neg_img, pos_img), axis=0)
        elif self.args.event_dim == 1:
            neg_img = np.sum(neg_volume, axis=0, keepdims=True)
            pos_img = np.sum(pos_volume, axis=0, keepdims=True)
            event_volume = neg_img + pos_img
        return event_volume

    def recursive_glob(self, rootdir='.', suffix=None):
        if isinstance(suffix, str):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames if filename.endswith(suffix)]
        elif isinstance(suffix, list):
            return [os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for x in suffix for filename in filenames if filename.startswith(x)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=(1024, 2048)),
            tr.ColorJitter(),
            tr.RandomGaussianBlur(),
            tr.RandomMotionBlur(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
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
    args.base_size = 1024
    args.crop_size = 512
    args.event_dim = 2

    cityscapes_train = BDD(args, split='train')
    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            event = sample['event'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='bdd')
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

