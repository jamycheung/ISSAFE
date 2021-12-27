import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import logging
import datetime
from dataloaders import make_data_loader
from dataloaders.utils import Colorize
from utils.metrics import Evaluator
from models.edcnet import EDCNet
import torch.backends.cudnn as cudnn


def write_config(args, log):
    log.info('\n----------------- Options ---------------\n')
    for k, v in sorted(vars(args).items()):
        log.info('{:>25}: {:<30}\n'.format(str(k), str(v)))
    log.info('----------------- End -------------------\n')


def create_logger(dir):
    logger = logging.getLogger("Logger")
    log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(dir, "eval_{}.log".format(log_time))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    hdlr.setLevel(logging.INFO)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


class Validator(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.time_train = []
        self.args.evaluate= True
        self.args.merge=True
        kwargs = {'num_workers':args.workers, 'pin_memory': False}
        _, self.val_loader, _, self.num_class = make_data_loader(args, **kwargs)
        print('un_classes:'+str(self.num_class))
        self.resize = args.crop_size if args.crop_size else [512, 1024]
        self.evaluator = Evaluator(self.num_class, self.logger)
        self.model = EDCNet(self.args.rgb_dim, args.event_dim, num_classes=self.num_class, use_bn=True)
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.to(self.args.device)
            cudnn.benchmark = True
        print('Model loaded successfully!')
        assert os.path.exists(args.weight_path), 'weight-path:{} doesn\'t exit!'.format(args.weight_path)
        self.new_state_dict = torch.load(os.path.join(args.weight_path), map_location='cuda:0')
        self.model = load_my_state_dict(self.model.module, self.new_state_dict['state_dict'])

    def validate(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, (sample, gt_path) in enumerate(tbar):
            target = sample['label']
            image = sample['image']
            event = sample['event']
            if self.args.cuda:
                target = target.to(self.args.device)
                image = image.to(self.args.device)
                event = event.to(self.args.device)
            start_time = time.time()
            with torch.no_grad():
                output, output_event = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)
            if self.args.cuda:
                torch.cuda.synchronize()
            if i!=0:
                fwt = time.time() - start_time
                self.time_train.append(fwt)
                print("Forward time per img (bath size=%d): %.3f (Mean: %.3f)" % (
                self.args.val_batch_size, fwt / self.args.val_batch_size,
                sum(self.time_train) / len(self.time_train) / self.args.val_batch_size))
            time.sleep(0.1) 

            pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())
            pre_colors_gt = Colorize()(torch.ByteTensor(target))
            checkname = self.args.weight_path.split('/')[-2]
            prediction_save_dir = os.path.join(self.args.label_save_path, checkname)
            if self.args.label_save:
                for j in range(pre_colors.shape[0]):
                    label_name = os.path.join(
                        *[prediction_save_dir, gt_path[j].split('gtFine/val/')[1].replace('/','_')])
                    if 'dada' in self.args.dataset:
                        label_name = label_name.replace('.jpg', '.png')
                    os.makedirs(os.path.dirname(label_name), exist_ok=True)
                    if 'dada' in self.args.dataset:
                        leftImg8bit_path = gt_path[j].replace('_labelTrainIds.png', '.jpg')
                    elif 'cityscape' in self.args.dataset:
                        leftImg8bit_path = gt_path[j].replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')
                    leftImg8bit_path = leftImg8bit_path.replace('/gtFine/', '/leftImg8bit/')
                    pre_color_image = ToPILImage()(pre_colors[j]) 
                    pre_colors_gt = ToPILImage()(pre_colors_gt[j])
                    img_ = Image.open(leftImg8bit_path)
                    img_ = img_.crop((280, 32, 1304, 544)) # [162, 0, 1422, 600]
                    img_ = img_.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
                    event_ = ToPILImage()(event[j].cpu())
                    pre_event_ = ToPILImage()(torch.sigmoid(output_event[j]).cpu()) # blur

                    if self.args.event_dim:
                        event_path = leftImg8bit_path.replace('/leftImg8bit/', '/event_image/')
                        event_path = event_path.replace('.jpg', '_event_image.png')
                        image_stack(img_, pre_color_image, pre_colors_gt, label_name, event_, pre_event_)
                    else:
                        image_stack(Image.open(leftImg8bit_path), pre_color_image, pre_colors_gt, label_name)
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.logger.info('Validation:')
        self.logger.info('[Epoch: %d, numImages: %5d]' % (0, i * self.args.batch_size + target.data.shape[0]))
        self.logger.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


def image_stack(org, pred, gt, savepath, event=None, pre_event_=None):
    imgs = [org, pred, gt] 
    if event is not None:
        event = event.convert('RGB')
        imgs.insert(1, event)
    if pre_event_ is not None:
        pre_event_ = pre_event_.convert('RGB')
        imgs.insert(2, pre_event_)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    store_img = np.vstack([np.asarray(i.resize(min_shape)) for i in imgs])
    store_img = Image.fromarray(store_img)
    store_img.save(savepath)


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Event validation")
    parser.add_argument('--model', type=str, default='EDCNet',
                        choices=['EDCNet'], help='model name (default: EDCNet)')
    parser.add_argument('--rgb-dim', type=int, default=3,
                        choices=[0, 3], help='whether use rgb as input (default: 3)')
    parser.add_argument('--event-dim', type=int, default=1,
                        choices=[1, 2, 18], help='event volume dimension (default: 2)')
    parser.add_argument('--dataset', type=str, default='dadaevent',
                        choices=['cityscapesevent', 'dadaevent', 'apolloscapeevent', 'bdd', 'kittievent', 'merge3'],
                        help='dataset name (default: dadaevent)')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=(512, 1024),
                        help='crop image size')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for validating (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--device', type=torch.device, default='cpu',
                        help='torch device')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--weight-path', type=str, default="./run/cityscapesevent/test_EDCNet_r18/model_best.pth",
                        help='enter your path of the weight')
    parser.add_argument('--label-save', action='store_true', default=False, help='save label')
    parser.add_argument('--label-save-path', type=str, default='results/',
                        help='path to save label')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.device = torch.device('cuda', args.gpu_ids[0])
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    log_dir = os.path.dirname(args.weight_path) + '/eval_DADA_log'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(log_dir)
    write_config(args, logger)
    validator = Validator(args, logger)
    validator.validate()


if __name__ == "__main__":
    main()