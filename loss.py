import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
from utils import *


def get_loss(args, cls_num_list, per_cls_weights):
    # Default Linear
    if args.loss_type == 'CE':
        criterion = CELoss(weight=per_cls_weights).cuda(
            args.gpu)  # nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
    elif args.loss_type == 'FeaBal':
        criterion = FeaBalLoss(cls_num_list=cls_num_list, weight=per_cls_weights, lambda_=args.lambda_).cuda(
            args.gpu)  # hyper-parameter A=60
    elif args.loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'GML':
        criterion = GML(cls_num_list).cuda(args.gpu)
    elif args.loss_type == 'LADE':
        criterion = LADELoss()
    elif args.loss_type == 'BSCE':
        criterion = BalancedSoftmax(cls_num_list)
    else:
        raise NotImplementedError(
            "Error:Loss function {} is not implemented! Please re-choose loss type!".format(args.loss_type))

    return criterion


class CELoss(nn.Module):
    def __init__(self, weight):
        super(CELoss, self).__init__()
        self.weight = weight

    def forward(self, out, labels, curr=0):
        """
        Args:
            out: dict out['feat'], embedding; out['score'], logit
            labels: ground truth labels with shape (batch_size).
        """
        feat, out = out['feature'], out['score']
        return F.cross_entropy(out, labels, weight=self.weight)

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)  # transfer to probability
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, curr=None):
        feat, input = input['feature'], input['score']
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        if (torch.cuda.is_available()):
            m_list = m_list.cuda()

        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, curr=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot

        index_float = index.type(torch.FloatTensor)
        if(torch.cuda.is_available()):
            index_float = index_float.cuda()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 取得对应位置的m   self.m_list
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)  # x的index位置换成x_m

        return F.cross_entropy(self.s * output, target, weight=self.weight)  # weight=self.weight


class FeaBalLoss(nn.Module):
    def __init__(self, cls_num_list, weight, lambda_=1., classifier=False, gamma=0.):
        super(FeaBalLoss, self).__init__()
        self.num_classes = len(cls_num_list)
        self.weight = weight
        self.classisier = classifier
        self.lambda_ = lambda_

        lam_list = torch.FloatTensor(cls_num_list)
        if (torch.cuda.is_available()):
            lam_list = lam_list.cuda()
        lam_list = torch.log(lam_list)  # s_list = s_list**(1/4)
        lam_list = lam_list.max() - lam_list
        self.lam_list = lam_list * (1 / lam_list.max())  # 归一化 lambda_：限制强度

        self.gamma = gamma

    def forward(self, out, labels, curr=0):
        """
        Args:
            out: dict out['feat'], embedding; out['score'], logit
            labels: ground truth labels with shape (batch_size).
        """
        feat, out = out['feature'], out['score']
        feat_norm = torch.norm(feat, dim=1).unsqueeze(1).repeat([1, len(self.lam_list)])

        logit = out - curr * self.lambda_ * self.lam_list / (feat_norm + 1e-12)

        if self.classisier:  # classifier re-balance model
            return F.cross_entropy(out, labels, weight=self.weight)
        else:
            return F.cross_entropy(logit, labels, weight=self.weight)

# 感谢作者

class GML(nn.Module):
    def __init__(self, num_class_list):
        super().__init__()
        self.p = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = torch.Tensor(num_class_list)
        if torch.cuda.is_available():
                self.weight = self.weight.cuda()

    def forward(self, output, target,curr=None):
        feat,output =  output['feature'],output['score']
        assert len(target.size()) == 1 # Target should be of 1-Dim

        max_logit = torch.max(output, dim=1, keepdim=True)[0] # of shape N x 1
        max_logit = max_logit.detach()
        logits = output - max_logit
        exp_logits = torch.exp(logits) * self.weight.view(-1, self.weight.shape[0])
        prob = torch.clamp(exp_logits / exp_logits.sum(1, keepdim=True), min=1e-5, max=1.)

        num_images, num_classes = prob.size()

        ground_class_prob = torch.gather(prob, dim=1, index=target.view(-1, 1))
        ground_class_prob = ground_class_prob.repeat((1, num_classes))

        mask = torch.zeros((num_images, num_classes), dtype=torch.int64, device=self.device)
        mask[range(num_images), target] = 1
        num_images_per_class = torch.sum(mask, dim=0)
        exist_class_mask = torch.zeros((num_classes,), dtype=torch.int64, device=self.device)
        exist_class_mask[num_images_per_class != 0] = 1

        num_images_per_class[num_images_per_class == 0] = 1 # avoid the dividing by zero exception

        mean_prob_classes = torch.sum(ground_class_prob * mask, dim=0) / num_images_per_class # of shape (C,)
        mean_prob_classes[exist_class_mask == 1] = -torch.log(mean_prob_classes[exist_class_mask == 1])

        mean_prob_sum = torch.sum(torch.pow(mean_prob_classes[exist_class_mask == 1], self.p)) / torch.sum(exist_class_mask)

        loss = torch.pow(mean_prob_sum, 1.0 / self.p)

        return loss

"""Copyright (c) Hyperconnect, Inc. and its affiliates.
All rights reserved.
"""

class LADELoss(nn.Module):
    def __init__(self, num_classes=10, img_max=500, prior=0.1, prior_txt=None, remine_lambda=0.1):
        super().__init__()
        if img_max is not None or prior_txt is not None:
            self.img_num_per_cls = calculate_prior(num_classes, img_max, prior, prior_txt, return_num=True).float()
            if torch.cuda.is_available():
                self.img_num_per_cls = self.img_num_per_cls.cuda()
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1. / num_classes).float()

        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float()))

        if torch.cuda.is_available():
            self.balanced_prior = self.balanced_prior.cuda()
            self.cls_weight = self.cls_weight.cuda()
    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None,curr=None):
        """
        y_pred: N x C
        target: N
        """
        y_pred = y_pred['score']
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

# import json


class BalancedSoftmax(nn.Module):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_class_list):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(num_class_list)

    def forward(self, input, label, reduction='mean',curr = None):
        input = input['score']
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)
