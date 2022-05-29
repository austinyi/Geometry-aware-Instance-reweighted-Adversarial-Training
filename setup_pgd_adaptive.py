"""
Setup PGD attack

source: https://github.com/wanglouis49/pytorch-adversarial_box
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


# Geometry-aware projected gradient descent (GA-PGD)
def GA_PGD(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output, target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, ep=None):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if torch.cuda.is_available():
            X_nat, y = X_nat.cuda(), y.cuda()

        if self.rand:
            rand = torch.Tensor(X_nat.shape).uniform_(-self.epsilon, self.epsilon)
            if torch.cuda.is_available():
                rand = rand.cuda()
            X = X_nat + rand
        else:
            X = X_nat.clone()

        y_var = to_var(y)

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad
            X += self.a * torch.sign(grad)

            diff = X - X_nat
            if ep is not None:
                new_diff = []
                for j in range(ep.shape[0]):
                    new_diff.append(torch.clamp(diff[j], -ep[j], ep[j]))
                new_diff = torch.stack(new_diff)
                diff = new_diff
            else:
                diff.clamp_(-self.epsilon, self.epsilon)

            X.detach().copy_((diff + X_nat).clamp_(0, 1))
        return X


def adv_train(X, y, model, criterion, adversary, ep=None):
    """
    Adversarial training. Returns pertubed mini batch.
    """

    # If adversarial training, need a snapshot of 
    # the model at each batch to compute grad, so 
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv = adversary.perturb(X, y, ep)

    return X_adv


def attack_over_test_data(model, adversary, param, loader_test, use_cuda=True, oracle=None):
    """
    Given target model computes accuracy on perturbed data
    """
    adversary.model = model
    total_correct = 0
    total_samples = len(loader_test.dataset)
    ntested = 0
    # For black-box
    if oracle is not None:
        total_samples -= param['hold_out_size']

    pbar = tqdm(loader_test)
    for X, y in pbar:
        y_pred = pred_batch(X, model)
        X_adv = adversary.perturb(X, y_pred)

        y_pred_adv = pred_batch(X_adv, model)
        ntested += y.size()[0]
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()
        pbar.set_postfix(adv_acc="{0}/{1} {2:-6.2f}%".format(total_correct, ntested,
                                                             total_correct * 100.0 / ntested),
                         refresh=False)
    pbar.close()
    acc = total_correct / total_samples
    print('Got %d/%d correct (%.2f%%) on the perturbed data'
          % (total_correct, total_samples, 100 * acc))

    return acc
