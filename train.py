import argparse
import os
import random
import time
import warnings
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from pathlib2 import Path

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.mlab as mlab

import models
from loss import get_loss
from dataset.dataset import get_dataset
from dataset.sampler import get_sampler_reweight
from utils import AverageMeter, mixup_criterion, mixup_data, prepare_folders, save_checkpoint, accuracy, \
    adjust_learning_rate, set_color, GM_HM_LR
from tqdm import tqdm


list_train_GM = []
list_train_Prec = []
list_train_HM = []
list_train_LR = []
list_test_GM = []
list_test_Prec = []
list_test_HM = []
list_test_LR = []

def main(args):
    store_name = prepare_folders(args)

    best_acc1, best_epoch = 0, 0

    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    '''
        Step 1: Data Loading
    '''
    train_dataset, val_dataset, cls_num_list = get_dataset(args)
    train_sampler, per_cls_weights = get_sampler_reweight(args.train_rule, train_dataset, args.gpu)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    '''
        Step 2: Create Model
    '''
    print("=> creating model '{}'".format(args.arch))
    classes_dict = {'cifar10': 10, 'cifar100': 100,}
    linear_dict = {'ce': 'Default', 'focal': 'Default', 'feabal': 'Default','gml':'Default','lade':'Default','bsce':'Default', \
                   'noise': 'Noise', \
                   'noiscr': 'Norm', 'noiang': 'Norm', 'ldam': 'Norm'}
    num_classes = classes_dict[args.dataset.lower()]

    classifier = True  # whether our network has classifier layer
    model = models.__dict__[args.arch](num_classes=num_classes, classifier=classifier,
                                       linear_type=linear_dict[args.loss_type.lower()], \
                                       pretrained=args.pretrained, pretrained_path=args.pretrained_path)
    # record old model
    oldmodel = model

    block = None
    classifier = None
    if torch.cuda.is_available():
        print("You are using GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    '''
        Step 3: Optimizer and Loss Setting
    '''
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = get_loss(args, cls_num_list, per_cls_weights)

    ''' 
        Step 4: Training Log 
    '''
    # init log
    tf_writer = SummaryWriter(log_dir=os.path.join(store_name, args.root_log))
    with open(os.path.join(store_name, args.root_log, 'args.txt'), 'w') as f:
        f.write(str(args))

    log_training = open(os.path.join(store_name, args.root_log, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(store_name, args.root_log, 'log_test.csv'), 'w')

    # save code
    code_dir = os.path.join(store_name, args.root_log, "codes")
    print("=> code will be saved in {}".format(code_dir))
    this_dir = Path.cwd()
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*spyproject*", "*.pth", "*log*", \
        "*checkpoint*", "*data*", "*result*", "*temp*", "saved"
    )
    shutil.copytree(this_dir, code_dir, ignore=ignore)

    '''
        Step 5: Resume Model
    '''
    if args.resume:  # the latest model checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:%d' % args.gpu)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    '''
        Step 6: Epochs Training
    '''



    for epoch in range(args.start_epoch, args.epochs):
        # Note that different dataset may have different decay strategy
        adjust_learning_rate(args, optimizer, epoch)
        train_one_epoch(args, train_loader, oldmodel, model, block, classifier, criterion, optimizer, epoch, log_training,
                        tf_writer, num_classes=num_classes)
        acc1, val_loss = validate_one_epoch(args, val_loader, oldmodel, model, block, classifier, criterion, epoch, log_testing,
                                            tf_writer, num_classes=num_classes)

        # scheduler.step(val_loss)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch

        save_checkpoint(store_name,
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'model_state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                        },
                        is_best)
        # record
        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f, Best epoch: %d\n' % (best_acc1, best_epoch)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

    '''
        Step 7: Drawing Figures
    '''

    # list_x = list(range(args.start_epoch, args.epochs))
    list_x_np = np.arange(args.start_epoch, args.epochs, 1)

    upper = 85
    lower = 65
    epoch_num = args.epochs - args.start_epoch
    array_upper = np.ones(epoch_num,dtype=int) * upper
    array_lower = np.ones(epoch_num,dtype=int) * lower
    list_train_Prec_np = np.array(list_train_Prec)
    supper = np.ma.masked_where(list_train_Prec_np < upper, list_train_Prec_np)
    slower = np.ma.masked_where(list_train_Prec_np > lower, list_train_Prec_np)
    smiddle = np.ma.masked_where((list_train_Prec_np < lower) | (list_train_Prec_np > upper), list_train_Prec_np)
    fig, axs = plt.subplots(1,2)


    axs[0].plot(list_x_np, smiddle, 'r-', list_x_np, slower, 'b-', list_x_np, supper, 'g-', list_x_np, array_upper, 'k--', list_x_np, array_lower, 'k--')
    axs[0].set_title("Train_Prec@1")
    axs[0].set_xlabel('Epoch')
    #axs[0].grid(True)

    os.makedirs('./results',exist_ok=True)

    upper = 60
    lower = 40
    array_upper = np.ones(epoch_num, dtype=int) * upper
    array_lower = np.ones(epoch_num, dtype=int) * lower
    list_test_Prec_np = np.array(list_test_Prec)
    supper = np.ma.masked_where(list_test_Prec_np < upper, list_test_Prec_np)
    slower = np.ma.masked_where(list_test_Prec_np > lower, list_test_Prec_np)
    smiddle = np.ma.masked_where((list_test_Prec_np < lower) | (list_test_Prec_np > upper), list_test_Prec_np)
    axs[1].plot(list_x_np, smiddle, 'r-', list_x_np, slower, 'b-', list_x_np, supper, 'g-', list_x_np, array_upper, 'k--', list_x_np, array_lower, 'k--')
    axs[1].set_title("Test_Prec@1")
    axs[1].set_xlabel('Epoch')
    #axs[1].grid(True)
    fig.set_size_inches(16, 5)
    plt.savefig('./results/Prec@1.png',dpi=1000)
    plt.clf()

    fig1, axs1 = plt.subplots(1, 2)
    list_train_GM_np = np.array(list_train_GM)
    axs1[0].plot(list_x_np, list_train_GM_np)
    axs1[0].set_title("Train_GM")
    axs1[0].set_xlabel('Epoch')
    axs1[0].grid(True)
    list_test_GM_np = np.array(list_test_GM)
    axs1[1].plot(list_x_np, list_test_GM_np)
    axs1[1].set_title("Test_GM")
    axs1[1].set_xlabel('Epoch')
    axs1[1].grid(True)
    fig1.set_size_inches(16, 5)
    plt.savefig('./results/GM.png', dpi=1000)
    plt.clf()

    fig2, axs2 = plt.subplots(1, 2)
    list_train_HM_np = np.array(list_train_HM)
    axs2[0].plot(list_x_np, list_train_HM_np)
    axs2[0].set_title("Train_HM")
    axs2[0].set_xlabel('Epoch')
    axs2[0].grid(True)
    list_test_HM_np = np.array(list_test_HM)
    axs2[1].plot(list_x_np, list_test_HM_np)
    axs2[1].set_title("Test_HM")
    axs2[1].set_xlabel('Epoch')
    axs2[1].grid(True)
    fig2.set_size_inches(16, 5)
    plt.savefig('./results/HM.png', dpi=1000)
    plt.clf()

    fig3, axs3 = plt.subplots(1, 2)
    list_train_LR_np = np.array(list_train_LR)
    axs3[0].plot(list_x_np, list_train_LR_np)
    axs3[0].set_title("Train_LR")
    axs3[0].set_xlabel('Epoch')
    axs3[0].grid(True)
    list_test_LR_np = np.array(list_test_LR)
    axs3[1].plot(list_x_np, list_test_LR_np)
    axs3[1].set_title("Test_LR")
    axs3[1].set_xlabel('Epoch')
    axs3[1].grid(True)
    fig3.set_size_inches(16, 5)
    plt.savefig('./results/LR.png', dpi=1000)
    plt.clf()
    return model
'''
    plt.plot(list_x, list_train_GM, color='blue', label='GM')
    plt.xlabel("Epochs")
    plt.title("Train_GM")
    plt.savefig('./Train_GM.png',dpi=1000)
    plt.clf()

    plt.plot(list_x, list_train_HM, color='green', label='HM')
    plt.xlabel("Epochs")
    plt.title("Train_HM")
    plt.savefig('./Train_HM.png',dpi=1000)
    plt.clf()

    plt.plot(list_x, list_train_LR, color='black', label='LR')
    plt.xlabel("Epochs")
    plt.title("Train_LR")
    plt.savefig('./Train_LR.png',dpi=1000)
    plt.clf()




    plt.plot(list_x, list_test_GM, color='blue', label='GM')
    plt.xlabel("Epochs")
    plt.title("Test_GM")
    plt.savefig('./Test_GM.png', dpi=1000)
    plt.clf()

    plt.plot(list_x, list_test_HM, color='green', label='HM')
    plt.xlabel("Epochs")
    plt.title("Test_HM")
    plt.savefig('./Test_HM.png', dpi=1000)
    plt.clf()

    plt.plot(list_x, list_test_LR, color='black', label='LR')
    plt.xlabel("Epochs")
    plt.title("Test_LR")
    plt.savefig('./Test_LR.png', dpi=1000)
    plt.clf()

    plt.plot(list_x, list_test_Prec, color='red', label='Prec@1')
    plt.xlabel("Epochs")
    plt.title("Test_Prec@1")
    plt.savefig('./Test_Prec@1.png', dpi=1000)
'''


def train_one_epoch(args, train_loader, oldmodel, model, block, classifier, criterion, optimizer, epoch, log, tf_writer, num_classes):  #
    all_batch_correct_per_class = torch.zeros(num_classes,dtype=torch.int64,requires_grad=False)
    all_batch_per_class = torch.zeros(num_classes,dtype=torch.int64,requires_grad=False)

    if torch.cuda.is_available():
        all_batch_correct_per_class = all_batch_correct_per_class.cuda()
        all_batch_per_class = all_batch_per_class.cuda()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()

    '''
    curriculum learning model
    '''
    # curr = (args.epochs-epoch)/args.epochs                 #linear. First tail, then head
    # curr = epoch/args.epochs                               #linear. First head, then tail
    curr = (epoch / (args.epochs - 10.1)) ** 2  # parabolic increase
    # curr = 1- math.cos(epoch / args.epochs * math.pi /2)   # cosine increase
    # curr = math.sin(epoch / args.epochs * math.pi /2)      # sine increase
    # curr = (1 - (epoch / args.epochs) ** 2) * 1            # parabolic increment
    # curr = np.random.beta(self.alpha, self.alpha)          # beta distribution

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader),
                                   desc=set_color(f"Train {epoch:>5}", 'pink')):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(input, target)
            output = model(images, get_feat=True)

            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam, curr=curr)

            output = output['score']

            output = output['score']
            output_old = oldmodel(input, get_feat=False) if args.pretrained and args.ensemble else None

            (acc1_a, acc5_a), batch_per_correct_a  = accuracy(output, output_old, args.t1, args.t2, targets_a, topk=(1, 5))
            (acc1_b, acc5_b), batch_per_correct_b = accuracy(output, output_old, args.t1, args.t2, targets_b, topk=(1, 5))
            acc1, acc5 = lam * acc1_a + (1 - lam) * acc1_b, lam * acc5_a + (1 - lam) * acc5_b
            batch_per_correct = lam * batch_per_correct + (1 - lam) * batch_per_correct
            all_batch_correct_per_class += batch_per_correct[0]
            all_batch_per_class += batch_per_correct[1]
        else:
            output = model(input, get_feat=True)

            loss = criterion(output, target, curr=curr)

            output = output['score']
            output_old = oldmodel(input, get_feat=False) if args.pretrained and args.ensemble else None

            (acc1, acc5),batch_per_correct = accuracy(output, output_old, args.t1, args.t2, target, topk=(1, 5))
            all_batch_correct_per_class += batch_per_correct[0]
            all_batch_per_class += batch_per_correct[1]

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    GM,HM,LR = GM_HM_LR(all_batch_correct_per_class,all_batch_per_class)

    list_train_GM.append(GM.item())
    list_train_HM.append(HM.item())
    list_train_LR.append(LR.item())
    list_train_Prec.append(top1.avg.item())

    output = ('Epoch [{0}/{1}]: lr: {lr:.5f}\t'
    # 'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    # 'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Total Loss: {loss.avg:.4f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}\t'
              'GM {GM:.3f}\t'
              'HM {HM:.3f}\t'
              'LR {LR:.3f}\n'.format(
        epoch, args.epochs,
        lr=optimizer.param_groups[-1]['lr'],
        loss=losses,
        top1=top1,
        top5=top5,
        GM=GM,
        HM=HM,
        LR=LR))

    print(output)
    if log is not None:
        log.write(output + '\n')
        # log.write(out_cls_acc + '\n')
        log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate_one_epoch(args, val_loader, oldmodel, model, block, classifier, criterion, epoch, log=None, tf_writer=None,
                       flag='Val',num_classes=100):
    all_batch_correct_per_class = torch.zeros(num_classes, dtype=torch.int64, requires_grad=False)
    all_batch_per_class = torch.zeros(num_classes, dtype=torch.int64, requires_grad=False)

    if torch.cuda.is_available():
        all_batch_correct_per_class = all_batch_correct_per_class.cuda()
        all_batch_per_class = all_batch_per_class.cuda()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader),
                                       desc=set_color(f"Test {epoch:>5}", 'pink')):
            if torch.cuda.is_available():
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input, get_feat=True)

            loss = criterion(output, target)

            output = output['score']
            output_old = oldmodel(input, get_feat=False) if args.pretrained and args.ensemble else None

            (acc1, acc5), batch_per_correct = accuracy(output, output_old, args.t1, args.t2, target, topk=(1, 5))
            all_batch_correct_per_class += batch_per_correct[0]
            all_batch_per_class += batch_per_correct[1]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        GM, HM, LR = GM_HM_LR(all_batch_correct_per_class, all_batch_per_class)
        # confusion_matrix
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        list_test_GM.append(GM.item())
        list_test_HM.append(HM.item())
        list_test_LR.append(LR.item())
        list_test_Prec.append(top1.avg.item())

        output = ('{flag} Results [{0}/{1}]: '
                  'Loss {loss.avg:.5f}\t'
                  'Prec@1 {top1.avg:.3f}\t'
                  'Prec@5 {top5.avg:.3f} '
                  'GM {GM:.3f}\t'
                  'HM {HM:.3f}\t'
                  'LR {LR:.3f}\n'
                  .format(epoch, args.epochs, flag=flag, loss=losses, top1=top1, top5=top5,GM=GM,HM=HM,LR=LR))
        # out_cls_acc = '%s Class Accuracy: %s'%(flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            # log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    # data_name = ['cifar10', 'cifar100', 'imagenet', 'inat', 'place365']
    # Places365; imagenet; iNaturelist2018
    # Place365: '/home/datasets/Places365'; imagenet: '/home/datasets/imagenet/ILSVRC2012_dataset';
    # iNaturelist2018: ' /home/datasets/iNaturelist2018'
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset setting')
    parser.add_argument('--data_path', default='./dataset/data_img', type=str, help='dataset setting')
    parser.add_argument('--img_path', default='/home/datasets/Places365', type=str, help='input image path')

    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
    parser.add_argument('--mixup', default=False, type=bool, help='if use mix-up')

    parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--root_log', type=str, default='./log')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    ## Please pay more attention to these parameters.
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32', help='model architecture')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--workers', '-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
    parser.add_argument('--exp_str', default='bs512_lr002_110', type=str,
                        help='number to indicate which experiment it is')  # bs128

    parser.add_argument('--loss_type', default="BSCE", type=str, help='loss type')  # LDAM CE FeaBal
    parser.add_argument('--imb_factor', default=0.01, type=float,
                        help='imbalance factor, Imbalance ratio: 0.01->100; 0.02->50')
    parser.add_argument('--batch_size', '-b', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--learning_rate', '--lr', default=0.1, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lambda_', default=60, type=float, metavar='N', help='the weight of A')
    parser.add_argument('--pretrained_path', default=None, type=str, help='the path of pretrained model')
    parser.add_argument('--ensemble', default=False, type=bool, help='ensemble old model and new model')
    parser.add_argument('--t1', default=1, type=int, metavar='N', help='new model temperature')
    parser.add_argument('--t2', default=1, type=int, metavar='N', help='old model temperature')
    ##
    args = parser.parse_args()

    # Training Config
    print("*******Training Config*******")
    for key, value in args._get_kwargs():
        if value is not None:
            print(key, '=', value)
    print('****************************')

    ## Main
    main(args)