import torch

def accuracy(output, target, return_length=False):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)

def top_k_acc(output, target, k=5, return_length=False):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)

def GM(output, target, return_length=False):
    with torch.no_grad():
        # print(target.size())
        batch_size = target.size(0)
        cls_nums = output.size(1)
        pred = torch.argmax(output,dim = 1)
        # print(pred.size())
        correct = pred.eq(target.expand_as(pred))
        one = torch.ones(batch_size)
        cls1 = torch.bincount(target,weights=correct,minlength=cls_nums).to(torch.float64)
        cls2 = torch.bincount(target,weights=one,minlength=cls_nums).to(torch.float64)
        if torch.cuda.is_available():
            cls1 = cls1.cuda()
            cls2 = cls2.cuda()
        cls2 = cls2.clamp(min=1e-3)
        acc_per_class = cls1 / cls2
        acc_per_class = acc_per_class.clamp(min=1e-3)
    if return_length:
        return torch.pow(torch.prod(acc_per_class),1/cls_nums), len(target)
    else:
        return torch.pow(torch.prod(acc_per_class),1/cls_nums)

def HM(output, target, return_length=False):
    with torch.no_grad():
        batch_size = target.size(0)
        cls_nums = output.size(1)
        pred = torch.argmax(output,dim = 1)
        correct = pred.eq(target.expand_as(pred))
        one = torch.ones(batch_size)
        cls1 = torch.bincount(target,weights=correct,minlength=cls_nums).to(torch.float64)
        cls2 = torch.bincount(target,weights=one,minlength=cls_nums).to(torch.float64)
        if torch.cuda.is_available():
            cls1 = cls1.cuda()
            cls2 = cls2.cuda()
        # Note that cls2 != 0,because of one epoch
        cls2 = cls2.clamp(min=1e-3)
        acc_per_class = cls1 / cls2
        acc_per_class = acc_per_class.clamp(min=1e-3)
    if return_length:
        return cls_nums/torch.sum(1/acc_per_class), len(target)
    else:
        return cls_nums/torch.sum(1/acc_per_class)

def LR(output, target, return_length=False):
    with torch.no_grad():
        batch_size = target.size(0)
        cls_nums = output.size(1)
        pred = torch.argmax(output,dim = 1)
        correct = pred.eq(target.expand_as(pred))
        one = torch.ones(batch_size)
        cls1 = torch.bincount(target,weights=correct,minlength=cls_nums).to(torch.float64)
        cls2 = torch.bincount(target,weights=one,minlength=cls_nums).to(torch.float64)
        if torch.cuda.is_available():
            cls1 = cls1.cuda()
            cls2 = cls2.cuda()
        # Note that cls2 != 0,because of one epoch
        cls2 = cls2.clamp(min=1e-3)
        acc_per_class = cls1/cls2
        acc_per_class = acc_per_class.clamp(min=1e-3)
    if return_length:
        return torch.min(acc_per_class), len(target)
    else:
        return torch.min(acc_per_class)

