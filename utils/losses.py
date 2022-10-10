import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)  \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def binary_ce_focal_loss(predict, target, gamma=2, alpha=0.25, reduction='mean', eps=1e-20):
    pt = torch.sigmoid(predict)  # sigmoide获取概率
    # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
    loss = - alpha * ((1 - pt).clamp(min=eps) ** gamma) * target * torch.log(pt.clamp(min=eps)) \
           - (1 - alpha) * (pt.clamp(min=eps) ** gamma) * (1 - target) * torch.log((1 - pt).clamp(min=eps))
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num)  #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor #),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def multi_ce_focal_loss(predict, target, class_num=2, gamma=2, alpha=None, reduction='mean', eps=1e-20):
    pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
    class_mask = F.one_hot(target, class_num)  # 获取target的one hot编码
    ids = target.view(-1, 1)
    if alpha is None:
        alpha = ids.new_ones(class_num, 1)
    alpha = alpha[ids.data.view(-1)].view(-1,1)  # 注意，这里的alpha是给定的一个list(tensor #),里面的元素分别是每一个类的权重因子
    probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
    log_p = probs.clamp(min=eps).log()  # 同样，原始ce上增加一个动态权重衰减因子
    loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss




def ghmc_loss(x, target, bins, alpha, last_bin_count=None):
    g = torch.abs(torch.sigmoid(x).detach() - target).detach()

    bin_idx = torch.floor(g * (bins - 0.0001)).long()

    bin_count = torch.zeros((bins))
    for i in range(bins):
        bin_count[i] = (bin_idx == i).sum().item()

    N = (x.size(0) * x.size(1))

    if last_bin_count is None:
        last_bin_count = bin_count
    else:
        bin_count = alpha * last_bin_count + (1 - alpha) * bin_count
        last_bin_count = bin_count

    nonempty_bins = (bin_count > 0).sum().item()

    gd = bin_count * nonempty_bins
    gd = torch.clamp(gd, min=0.0001)
    beta = N / gd

    return F.binary_cross_entropy_with_logits(x, target, weight=beta[bin_idx]), last_bin_count


class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    # 分类损失
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    # 回归损失
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)