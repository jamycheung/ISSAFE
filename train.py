"""
train EDCNet
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from mypath import Path
from dataloaders import make_data_loader
from models.edcnet import EDCNet
from utils.loss import SegmentationLosses
from models.replicate import patch_replication_callback
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.logger = self.saver.create_logger()

        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.model = EDCNet(args.rgb_dim, args.event_dim, num_classes=self.nclass, use_bn=True)
        train_params = [{'params': self.model.random_init_params(),
                         'lr': 10*args.lr, 'weight_decay': 10*args.weight_decay},
                        {'params': self.model.fine_tune_params(),
                         'lr': args.lr, 'weight_decay': args.weight_decay}]
        self.optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.to(self.args.device)
        if args.use_balanced_weights:
            root_dir = Path.db_root_dir(args.dataset)[0] if isinstance(Path.db_root_dir(args.dataset), list) else Path.db_root_dir(args.dataset)
            classes_weights_path = os.path.join(root_dir,
                                                args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass, classes_weights_path)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion_event = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='event')
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader), warmup_epochs=5)

        self.evaluator = Evaluator(self.nclass, self.logger)
        self.saver.save_model_summary(self.model)
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            target = sample['label']
            image = sample['image']
            event = sample['event']
            if self.args.cuda:
                target = target.to(self.args.device)
                image = image.to(self.args.device)
                event = event.to(self.args.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output, output_event = self.model(image)
            loss = self.criterion(output, target)
            loss_event = self.criterion_event(output_event, event)
            loss += (loss_event * 0.1)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss/num_img_tr, epoch)
        self.logger.info('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + target.data.shape[0]))
        self.logger.info('Loss: %.3f' % (train_loss/num_img_tr))

        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        num_img_val = len(self.val_loader)
        for i, (sample, _) in enumerate(tbar):
            target = sample['label']
            image = sample['image']
            event = sample['event']
            if self.args.cuda:
                target = target.to(self.args.device)
                image = image.to(self.args.device)
                event = event.to(self.args.device)
            with torch.no_grad():
                output, output_event = self.model(image)
            loss = self.criterion(output, target)
            loss_event = self.criterion_event(output_event, event)
            loss += (loss_event * 20)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss/len(self.val_loader), epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.logger.info('Validation:')
        self.logger.info('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + target.data.shape[0]))
        self.logger.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        self.logger.info('Loss: %.3f' % (test_loss/num_img_val))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Event Training")
    parser.add_argument('--model', type=str, default='EDCNet',
                        choices=['EDCNet'], help='model name (default: EDCNet)')
    parser.add_argument('--rgb-dim', type=int, default=3,
                        choices=[0, 3], help='whether use rgb as input (default: 3)')
    parser.add_argument('--event-dim', type=int, default=2,
                        choices=[1, 2, 18], help='event volume dimension (default: 2)')
    parser.add_argument('--dataset', type=str, default='cityscapesevent',
                        choices=['cityscapesevent', 'dadaevent', 'apolloscapeevent', 'bdd', 'kittievent', 'merge3'],
                        help='dataset name (default: cityscapesevent)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset when pascal (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=(512, 1024),
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['ce', 'focal', 'ohem'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=2,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: True)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: cos)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2.5e-5,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--device', type=torch.device, default='cpu',
                        help='torch device')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='test_EDCNet_r18',
                        help='set the checkpoint name, folder to store output')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=2,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.device = torch.device('cuda', args.gpu_ids[0])
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.epochs is None:
        epoches = {
            'cityscapesevent': 200,
            'dadaevent': 200,
            'apolloscapeevent': 100
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'cityscapesevent': 0.0001,
            'dadaevent': 0.0001,
            'apolloscapeevent': 0.0001
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'EDCNet'

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
