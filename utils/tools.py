import os, re
import time
import random

import numpy as np
import calendar
import time
import csv

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg

def create_target_folders(root_path, folder_name, target_name, timestamp, split_files=False):
    folder_path = os.path.join(root_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    # dest_path = os.path.join(folder_path, config.ALL_SOURCES_FOLDER) if train_source else 
    dest_path = os.path.join(folder_path, str(target_mapping(target_name)) + "-" +target_name)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    target_files_path = os.path.join(dest_path, "splitfiles" if split_files else "files")#files, splitfiles
    if not os.path.exists(target_files_path):
        os.makedirs(target_files_path, exist_ok=True)

    if timestamp is None:
        timestamp = calendar.timegm(time.gmtime())
        target_timestamp_path = os.path.join(dest_path, str(timestamp))
        os.makedirs(target_timestamp_path, exist_ok=True)
    else:
        target_timestamp_path = os.path.join(dest_path, str(timestamp))
    target_weights_path = os.path.join(target_timestamp_path, "weights")
    if not os.path.exists(target_weights_path):
        os.makedirs(target_weights_path, exist_ok=True)
    
    return target_files_path, target_weights_path, str(timestamp)

def target_mapping(target_name):
    if target_name == "081014_w_27":
        return 1
    elif target_name == "101609_m_36":
        return 2
    elif target_name == "112009_w_43":
        return 3
    elif target_name == "091809_w_43":
        return 4
    elif target_name == "071309_w_21":
        return 5
    elif target_name == "073114_m_25":
        return 6
    elif target_name == "080314_w_25":
        return 7
    elif target_name == "073109_w_28":
        return 8
    elif target_name == "100909_w_65":
        return 9
    elif target_name == "081609_w_40":
        return 10
    else:
        return 0

class MetricLogger:
    def __init__(self, save_dir="metrics", filename="metrics_log.csv"):
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, filename)

        # Write header once
        with open(self.filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "model", "au_sim_mean", "au_sim_std", "logits_mean", "logits_std"])

    def log(self, epoch, model_name, au_sim, logits):
        """Log metrics for each epoch."""
        au_sim_mean = au_sim.mean().item()
        au_sim_std = au_sim.std().item()
        logits_mean = logits.mean().item()
        logits_std = logits.std().item()

        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model_name, au_sim_mean, au_sim_std, logits_mean, logits_std])

        print(f"[Metrics] {model_name} | Epoch {epoch:03d} | "
              f"AU μ={au_sim_mean:.4f}, σ={au_sim_std:.4f}, "
              f"Logit μ={logits_mean:.4f}, σ={logits_std:.4f}")

def load_bah_src_subs(file_path, topk=None, selected_sub_list=None, index2value=False):
    subject_list = None
    unique_ids = extract_unique_ids(file_path)
    if topk is None:
        subject_list = unique_ids
        # subject_list = ['82553', '82554', '82555', '82557', '82563','82564', '82565', '82565', # 7
        #                  # 15    
        #                 ] # 76
    # 071709_w_23, 101814_m_58,080609_w_27,102008_w_22, 112310_m_20,112809_w_23,102316_w_50,071814_w_23,102214_w_36 // ontsne
    if selected_sub_list is not None:
        if index2value:
            return [subject_list[i] for i in selected_sub_list]
        else:
            # return [sub for sub in subject_list if sub in selected_sub_list]
            return [subject_list.index(sub) for sub in selected_sub_list if sub in subject_list]
    return subject_list

def extract_unique_ids(file_path):
    # Read file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract all IDs that come after 'Videos/' using regex
    ids = re.findall(r'Videos/(\d+)/', content)

    # Get unique IDs
    unique_ids = sorted(set(ids))

    # Print results
    print("Total unique ID count:", len(unique_ids))
    # print("Unique ID list:")
    # print(unique_ids)

    return unique_ids