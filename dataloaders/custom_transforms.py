import torch
import random
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['label'] = np.array(sample['label']).astype(np.float32)
        if 'image' in sample:
            img = sample['image']
            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            sample['image'] = img
        if 'event' in sample:
            event = sample['event']
            event = event.astype(np.float32)
            # binary
            event[event > 0] = 1
            sample['event'] = event
        return sample


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['label'] = torch.from_numpy(np.array(sample['label']).astype(np.float32)).float()
        if 'image' in sample:
            img = sample['image']
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            sample['image'] = img
        if 'event' in sample:
            sample['event'] = torch.from_numpy(np.array(sample['event']).astype(np.float32)).float()

        return sample

class CropBlackArea(object):
    """
    crop black area for event image
    """
    def __init__(self, crop=None):
        self.crop = [140, 30, 2030, 900] if crop is None else crop
    def __call__(self, sample):
        width, height = sample['label'].size
        left, top, right, bottom = self.crop
        sample['label'] = sample['label'].\
            crop((left, top, right, bottom)).\
            resize((width,height), Image.NEAREST)
        if 'image' in sample:
            img = sample['image']
            img = img.crop((left, top, right, bottom))
            img = img.resize((width, height), Image.BILINEAR)
            sample['image'] = img
        if 'event' in sample:
            event = sample['event']
            event = event[:, top:bottom, left:right]
            event = torch.unsqueeze(torch.from_numpy(event).float(), 0)
            event = torch.nn.functional.interpolate(event, size=[height, width], mode="nearest")
            event = event[0].cpu().numpy()
            sample['event'] = event

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['label'] = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)
            if 'image' in sample:
                img = sample['image']
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                sample['image'] = img
            if 'event' in sample:
                event = sample['event']
                event = np.flip(event, axis=2)
                sample['event'] = event
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            img = sample['image']
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            sample['image'] = img
        return sample

class RandomMotionBlur():
    def __call__(self, sample):
        if random.random() < 0.5:
            img = sample['image']
            degree = random.randint(1, 15)
            angle = random.randint(1, 45)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
            motion_blur_kernel = motion_blur_kernel / degree
            img = np.array(img)
            blurred = cv2.filter2D(img, -1, motion_blur_kernel)

            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            blurred = Image.fromarray(blurred.astype(np.uint8))
            sample['image'] = blurred
        return sample


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, sample):
        img = sample['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        sample['image'] = img
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.base_size = base_size  # 1024
        self.fill = fill

    def __call__(self, sample):
        w, h = sample['label'].size
        short_edge = w if w<h else h
        self.base_size = self.base_size if self.base_size<short_edge else short_edge
        # self.crop_size = self.crop_size if self.crop_size<short_edge else int(short_edge*0.75)
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.25), int(self.base_size * 2.0))
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        if 'image' in sample:
            sample['image'] = sample['image'].resize((ow, oh), Image.BILINEAR)
        if 'event' in sample:
            event = sample['event']
            event = torch.unsqueeze(torch.from_numpy(event.copy()).float(), 0)
            event = torch.nn.functional.interpolate(event, size=[oh, ow], mode="nearest")
            event = event[0].cpu().numpy()
            sample['event'] = event
        mask = sample['label']
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < min(self.crop_size):
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            if 'image' in sample:
                sample['image'] = ImageOps.expand(sample['image'], border=(0, 0, padw, padh),
                                                  fill=0)#left,top,right,bottom
            if 'event' in sample:
                sample['event'] = np.pad(sample['event'], pad_width=((0, 0), (0, padh), (0, padw)),
                                         mode='constant', constant_values=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = mask.size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        if 'image' in sample:
            sample['image'] = sample['image'].crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        if 'event' in sample:
            sample['event'] = sample['event'][:, y1:y1 + self.crop_size[0], x1:x1 + self.crop_size[1]]
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        sample['label'] = mask
        return sample


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        event = sample['event']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        event = event.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        event = event.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'event': event,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        sample['label'] = sample['label'].resize((self.size[1], self.size[0]), Image.NEAREST)
        if 'image' in sample:
            sample['image'] = sample['image'].resize((self.size[1], self.size[0]), Image.BILINEAR)
        if 'event' in sample:
            event = sample['event']
            event = torch.unsqueeze(torch.from_numpy(event.copy()).float(), 0)
            event = torch.nn.functional.interpolate(event, size=self.size, mode="nearest")
            event = event[0].cpu().numpy()
            sample['event'] = event
        return sample


class TranslationXY(object):
    def __call__(self, sample):
        transX = random.randint(-2, 2)
        transY = random.randint(-2, 2)
        mask = sample['label']
        mask = ImageOps.expand(mask, border=(transX, transY, 0, 0), fill=255)  # pad label filling with 255
        mask = mask.crop((0, 0, mask.size[0] - transX, mask.size[1] - transY))
        sample['label'] = mask

        if 'image' in sample:
            img = sample['image']
            img = ImageOps.expand(img, border=(transX, transY, 0, 0), fill=0)
            img = img.crop((0, 0, img.size[0] - transX, img.size[1] - transY))
            sample['image'] = img
        if 'event' in sample:
            event = sample['event']
            event = np.pad(event, pad_width=((0, 0), (transY, 0), (transX, 0)), mode='constant', constant_values=0)
            event = event[:, 0:event.size[1] - transY, 0:event.size[0] - transX]
            sample['event'] = event
        return sample


if __name__ == '__main__':
    from PIL import Image
    mb = RandomMotionBlur()
    img = Image.open('kitti_rgb.png')
    sample = {'image': img}
    img_ = mb(sample)

    img.show(title='Source image')
    img_['image'].show(title='blur image')
