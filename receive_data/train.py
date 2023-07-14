from __future__ import print_function
import os
from util.resample import SpeedPerturb
# from models.resnet56_moe_debug import  resnet56, L1_loss
from config.config_origin import Config
from models.cnn import resnet20
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import time
from sklearn import preprocessing

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    res = np.sum([p.numel() for p in model.parameters()]).item()
    return res / 1024 / 1024


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

class MyDataset(Dataset):
    def __init__(self, file_dir):
        self.imgs = []
        labels = []
        for root, sub_folders, files in os.walk(file_dir):
            for name in files:
                self.imgs.append(os.path.join(root, name))
                labels.append(root.split("\\")[1])
        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(labels)


    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.targets[index]
        img = np.loadtxt(img, delimiter=',', dtype='float')
        img = np.transpose(img, (1, 0))
        return img, label

    def __len__(self):
        return len(self.imgs)

def get_data():
    dataset = MyDataset("data")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=True)

    return train_loader, test_loader



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data_input, label = data
        data_input = data_input.float().to(device)
        data_input, aug_lens = data_aug(data_input)

        label = label.to(device)
        label = torch.cat([label]*aug_lens)






        output = model(data_input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.float().data, label)[0]
        losses.update(loss.float().item(), data[0].size(0))
        top1.update(prec1.item(), data[0].size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def val(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_input, label = data
            data_input = data_input.float().to(device)

            label = label.to(device)
            output = model(data_input)
            loss = criterion(output, label)


            prec1 = accuracy(output.float().data, label)[0]
            losses.update(loss.float().item(), data[0].size(0))
            top1.update(prec1.item(), data[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg


def data_aug(wav):
    wavs_aug_tot = []
    wavs_aug_tot.append(wav)
    aug_list=[70, 140]

    for i in aug_list:
        speed_perturb = SpeedPerturb(
            perturb_prob=1.0, orig_freq=3000, speeds=[i]
        )


        wavs_aug = speed_perturb(wav)

        if wavs_aug.shape[1] > wav.shape[1]:
            wavs_aug = wavs_aug[:, 0: wav.shape[1]]
        else:
            zero_sig = torch.zeros_like(wav)
            zero_sig[:, 0: wavs_aug.shape[1]] = wavs_aug
            wavs_aug = zero_sig
        wavs_aug_tot.append(wavs_aug)
    wavs = torch.cat(wavs_aug_tot, dim=0)
    lens = len(aug_list)+1

    return wavs, lens



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    global best_prec1
    best_prec1 = 0
    opt = Config()
    device = torch.device("cuda:0" if opt.USE_CUDA else "cpu")

    train_loader, test_loader = get_data()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)





    model = resnet20()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr)

    start_epoch = 0
    if opt.RESUME:
        print("loading best checkpoint...")

        checkpoint = torch.load(opt.path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # scheduler.load_state_dict(checkpoint['scheduler'])

    start = time.time()

    for epoch in range(start_epoch, opt.max_epoch):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        # scheduler.step()
        prec1 = val(test_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % opt.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(opt.save_dir, 'checkpoint.th'))

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(opt.save_dir, 'model.th'))

            with open(opt.save_dir + "/" + opt.backbone + "_" + opt.dataset + "_best", "a")as f:
                f.write("Epoch: " + str(epoch) + "\t Prec@1 " + str(best_prec1) + "\n")
                f.close()
