import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False): # ignore_index=255
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'event':
            return self.BinaryCrossEntropyLoss
        elif mode == 'ohem':
            return self.OHEMCELoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.to(torch.device('cuda', target.get_device()))

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.to(torch.device('cuda', target.get_device()))

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def OHEMCELoss(self, logit, target, thresh=0.7):
        #target = target.long()
        n_min = target[target != 255].numel() // 16
        thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        loss = criterion(logit, target.long()).view(-1)
        loss_hard = loss[loss > thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

    def BinaryCrossEntropyLoss(self, logit, target):
        """event binary"""
        n, c, h, w = logit.size()
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()  # = sigmoid and BCELoss
        if self.cuda:
            criterion = criterion.to(torch.device('cuda', target.get_device()))
        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n
        #print(logit.shape, target.shape, loss.shape)
        return loss

# class CrossEntropyLoss2d(torch.nn.Module):
#
#     def __init__(self, weight=None):
#         super().__init__()
#
#         self.loss = torch.nn.NLLLoss2d(weight)
#
#     def forward(self, outputs, targets):
#         return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets.long())

class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, cuda=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.cuda = cuda
    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        self.nll_loss = nn.CrossEntropyLoss(weight=self.weight,ignore_index=self.ignore_index,reduction='mean')
        if self.cuda:
            self.nll_loss = self.nll_loss.cuda()
        return self.nll_loss(output, target.long())

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




