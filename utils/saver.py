import os
import shutil
import torch
from collections import OrderedDict
import glob
# from torchsummary import summary
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
import datetime

from collections import OrderedDict
import numpy as np


def summary(model, input_size, f, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            if input[0] is not None:
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x= []
    for in_size in input_size:
        if in_size is None:
            x.append(None)
        else:
            x.append(torch.rand(2, *in_size).type(dtype))
    # x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    f.write("----------------------------------------------------------------\n")
    line_new = "{:>20}  {:>25} {:>15}\n".format("Layer (type)", "Output Shape", "Param #")
    f.write(line_new)
    f.write("================================================================\n")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}\n".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod([np.prod(outshape) for outshape in summary[layer]["output_shape"]])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        f.write(line_new)

    # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_input_size = abs(np.sum([np.prod(in_tuple) for in_tuple in input_size if in_tuple is not None]) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    f.write("================================================================\n")
    f.write("Total params: {0:,}\n".format(total_params))
    f.write("Trainable params: {0:,}\n".format(trainable_params))
    f.write("Non-trainable params: {0:,}\n".format(total_params - trainable_params))
    f.write("----------------------------------------------------------------\n")
    f.write("Input size (MB): %0.2f\n" % total_input_size)
    f.write("Forward/backward pass size (MB): %0.2f\n" % total_output_size)
    f.write("Params size (MB): %0.2f\n" % total_params_size)
    f.write("Estimated Total Size (MB): %0.2f\n" % total_size)
    f.write("----------------------------------------------------------------")
    # return summary

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(*['run', args.dataset, args.checkname])
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):  # filename from .pth.tar change to .pth?
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.args).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        # print(message)
        with open(logfile, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def save_model_summary(self, model, torchsummary=False):
        if torchsummary:
            modelfile = os.path.join(self.experiment_dir, 'model_torchsummary.txt')
            with open(modelfile, 'w') as f:
                event_shape = (9, self.args.event_dim, 512, 256) if self.args.model=='STNet' else (self.args.event_dim, 512, 256)
                rgb_shape = (self.args.rgb_dim, 512, 256)
                if self.args.event_dim!=0 and self.args.rgb_dim!=0:
                    summary(model, [rgb_shape, event_shape], f)
                elif self.args.event_dim!=0 and self.args.rgb_dim==0:
                    summary(model, [None, event_shape], f)
                elif self.args.event_dim==0 and self.args.rgb_dim!=0:
                    summary(model, [rgb_shape, None], f)
        modelfile = os.path.join(self.experiment_dir, 'model_functionsummary.txt')
        with open(modelfile, 'w') as f:
            f.write(str(model))

    def create_logger(self):
        logger = logging.getLogger("Logger")
        log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.experiment_dir, "train_{}.log".format(log_time))
        hdlr = logging.FileHandler(file_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        hdlr.setLevel(logging.INFO)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger