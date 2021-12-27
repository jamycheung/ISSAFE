import numpy as np


class Evaluator(object):
    def __init__(self, num_class, log=None):
        self.logger = log
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        self.logger.info('-----------Acc of each classes-----------')
        self.logger.info("road         : %.6f" % (Acc[0] * 100.0))
        self.logger.info("sidewalk     : %.6f" % (Acc[1] * 100.0))
        self.logger.info("building     : %.6f" % (Acc[2] * 100.0))
        self.logger.info("wall         : %.6f" % (Acc[3] * 100.0))
        self.logger.info("fence        : %.6f" % (Acc[4] * 100.0))
        self.logger.info("pole         : %.6f" % (Acc[5] * 100.0))
        self.logger.info("traffic light: %.6f" % (Acc[6] * 100.0))
        self.logger.info("traffic sign : %.6f" % (Acc[7] * 100.0))
        self.logger.info("vegetation   : %.6f" % (Acc[8] * 100.0))
        if self.num_class == 19:
            self.logger.info("terrain      : %.6f" % (Acc[9] * 100.0))
            self.logger.info("sky          : %.6f" % (Acc[10] * 100.0))
            self.logger.info("person       : %.6f" % (Acc[11] * 100.0))
            self.logger.info("rider        : %.6f" % (Acc[12] * 100.0))
            self.logger.info("car          : %.6f" % (Acc[13] * 100.0))
            self.logger.info("truck        : %.6f" % (Acc[14] * 100.0))
            self.logger.info("bus          : %.6f" % (Acc[15] * 100.0))
            self.logger.info("train        : %.6f" % (Acc[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (Acc[17] * 100.0))
            self.logger.info("bicycle      : %.6f" % (Acc[18] * 100.0))
        elif self.num_class == 16:
            # self.logger.info("terrain      : %.6f" % (Acc[9] * 100.0))
            # self.logger.info("sky          : %.6f" % (Acc[10] * 100.0))
            self.logger.info("person       : %.6f" % (Acc[9] * 100.0))
            self.logger.info("rider        : %.6f" % (Acc[10] * 100.0))
            self.logger.info("car          : %.6f" % (Acc[11] * 100.0))
            self.logger.info("truck        : %.6f" % (Acc[12] * 100.0))
            self.logger.info("bus          : %.6f" % (Acc[13] * 100.0))
            # self.logger.info("train        : %.6f" % (Acc[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (Acc[14] * 100.0))
            self.logger.info("bicycle      : %.6f" % (Acc[15] * 100.0))

        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        self.logger.info('-----------IoU of each classes-----------')
        self.logger.info("road         : %.6f" % (MIoU[0] * 100.0))
        self.logger.info("sidewalk     : %.6f" % (MIoU[1] * 100.0))
        self.logger.info("building     : %.6f" % (MIoU[2] * 100.0))
        self.logger.info("wall         : %.6f" % (MIoU[3] * 100.0))
        self.logger.info("fence        : %.6f" % (MIoU[4] * 100.0))
        self.logger.info("pole         : %.6f" % (MIoU[5] * 100.0))
        self.logger.info("traffic light: %.6f" % (MIoU[6] * 100.0))
        self.logger.info("traffic sign : %.6f" % (MIoU[7] * 100.0))
        self.logger.info("vegetation   : %.6f" % (MIoU[8] * 100.0))
        if self.num_class == 19:
            self.logger.info("terrain      : %.6f" % (MIoU[9] * 100.0))
            self.logger.info("sky          : %.6f" % (MIoU[10] * 100.0))
            self.logger.info("person       : %.6f" % (MIoU[11] * 100.0))
            self.logger.info("rider        : %.6f" % (MIoU[12] * 100.0))
            self.logger.info("car          : %.6f" % (MIoU[13] * 100.0))
            self.logger.info("truck        : %.6f" % (MIoU[14] * 100.0))
            self.logger.info("bus          : %.6f" % (MIoU[15] * 100.0))
            self.logger.info("train        : %.6f" % (MIoU[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (MIoU[17] * 100.0))
            self.logger.info("bicycle      : %.6f" % (MIoU[18] * 100.0))
        elif self.num_class == 16:
            # self.logger.info("terrain      : %.6f" % (MIoU[9] * 100.0))
            # self.logger.info("sky          : %.6f" % (MIoU[10] * 100.0))
            self.logger.info("person       : %.6f" % (MIoU[9] * 100.0))
            self.logger.info("rider        : %.6f" % (MIoU[10] * 100.0))
            self.logger.info("car          : %.6f" % (MIoU[11] * 100.0))
            self.logger.info("truck        : %.6f" % (MIoU[12] * 100.0))
            self.logger.info("bus          : %.6f" % (MIoU[13] * 100.0))
            # self.logger.info("train        : %.6f" % (MIoU[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (MIoU[14] * 100.0))
            self.logger.info("bicycle      : %.6f" % (MIoU[15] * 100.0))

        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print frequence of each class
        self.logger.info('-----------Freq of each classes-----------')
        self.logger.info("road         : %.6f" % (freq[0] * 100.0))
        self.logger.info("sidewalk     : %.6f" % (freq[1] * 100.0))
        self.logger.info("building     : %.6f" % (freq[2] * 100.0))
        self.logger.info("wall         : %.6f" % (freq[3] * 100.0))
        self.logger.info("fence        : %.6f" % (freq[4] * 100.0))
        self.logger.info("pole         : %.6f" % (freq[5] * 100.0))
        self.logger.info("traffic light: %.6f" % (freq[6] * 100.0))
        self.logger.info("traffic sign : %.6f" % (freq[7] * 100.0))
        self.logger.info("vegetation   : %.6f" % (freq[8] * 100.0))
        if self.num_class == 19:
            self.logger.info("terrain      : %.6f" % (freq[9] * 100.0))
            self.logger.info("sky          : %.6f" % (freq[10] * 100.0))
            self.logger.info("person       : %.6f" % (freq[11] * 100.0))
            self.logger.info("rider        : %.6f" % (freq[12] * 100.0))
            self.logger.info("car          : %.6f" % (freq[13] * 100.0))
            self.logger.info("truck        : %.6f" % (freq[14] * 100.0))
            self.logger.info("bus          : %.6f" % (freq[15] * 100.0))
            self.logger.info("train        : %.6f" % (freq[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (freq[17] * 100.0))
            self.logger.info("bicycle      : %.6f" % (freq[18] * 100.0))
        elif self.num_class == 16:
            # self.logger.info("terrain      : %.6f" % (freq[9] * 100.0))
            # self.logger.info("sky          : %.6f" % (freq[10] * 100.0))
            self.logger.info("person       : %.6f" % (freq[9] * 100.0))
            self.logger.info("rider        : %.6f" % (freq[10] * 100.0))
            self.logger.info("car          : %.6f" % (freq[11] * 100.0))
            self.logger.info("truck        : %.6f" % (freq[12] * 100.0))
            self.logger.info("bus          : %.6f" % (freq[13] * 100.0))
            # self.logger.info("train        : %.6f" % (freq[16] * 100.0))
            self.logger.info("motorcycle   : %.6f" % (freq[14] * 100.0))
            self.logger.info("bicycle      : %.6f" % (freq[15] * 100.0))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




