import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from GAIR import GAIR
import numpy as np
import attack_generator as attack
from utils import Logger
from setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data
from tqdm import tqdm



# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch):
    Lam = float(args.Lambda)
    if args.epochs >= 110:
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch / args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam - 1.0
            elif epoch >= 110:
                Lambda = Lam - 1.5
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch / args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam - 2.0
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda


def trainClassifier(args, model, train_loader, test_loader, use_cuda=True):
    if use_cuda:
        model = model.cuda()
        # model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    train_criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # training
        ave_loss = 0
        step = 0
        # Get lambda
        Lambda = adjust_Lambda(epoch + 1)
        # num_data = 0
        # train_robust_loss = 0
        for idx, (x, target) in enumerate(train_loader):
            x, target = to_var(x), to_var(target)

            x_adv, Kappa = attack.GA_PGD(model, x, target, args.epsilon, args.step_size, args.num_steps,
                                         loss_fn="cent",
                                         category="Madry", rand_init=True)
                
            model.train()
            lr = lr_schedule(epoch + 1)
            optimizer.param_groups[0].update(lr=lr)
            optimizer.zero_grad()

            if (epoch + 1) >= args.begin_epoch:
                Kappa = Kappa.cuda()
                loss = train_criterion(model(x_adv), target)
                # Calculate weight assignment according to geometry value
                normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
                loss = loss.mul(normalized_reweight).mean()
            else:
                loss = train_criterion(model(x_adv), target)

            loss.backward()
            optimizer.step()
            
        acc = testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=100)
        print("Epoch {} test accuracy: {:.3f}".format(epoch, acc))
    return model

def testClassifier(test_loader, model, use_cuda=True, batch_size=100):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc


def testattack(classifier, test_loader, args, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=args.epsilon, k=args.num_steps, a=args.step_size)
    param = {
    'test_batch_size': 100,
    'epsilon': args.epsilon,
    }
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc



def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')

    # Setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    if args.net == "WRN":
        model = Wide_ResNet_Madry(depth=args.depth, num_classes=10, widen_factor=args.width_factor,
                                  dropRate=args.drop_rate).cuda()

    print('==> Training starts..')
    model = trainClassifier(args, model, train_loader, test_loader, use_cuda=use_cuda)
    testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=100)
    testattack(model, test_loader, args, use_cuda=use_cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAIRAT: Geometry-aware instance-dependent adversarial training')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
    parser.add_argument('--num-steps', type=int, default=7, help='maximum perturbation step K')
    parser.add_argument('--step-size', type=float, default=0.007, help='step size')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--net', type=str, default="WRN",
                        help="decide which network to use,choose from smallcnn,resnet18,WRN")
    parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
    parser.add_argument('--random', type=bool, default=True, help="whether to initiat adversarial sample with random noise")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width-factor', type=int, default=10, help='WRN width factor')
    parser.add_argument('--drop-rate', type=float, default=0.0, help='WRN drop rate')
    parser.add_argument('--resume', type=str, default=None, help='whether to resume training')
    parser.add_argument('--out-dir', type=str, default='./GAIRAT_result', help='dir of output')
    parser.add_argument('--lr-schedule', default='piecewise',
                        choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--Lambda', type=str, default='-1.0', help='parameter for GAIR')
    parser.add_argument('--Lambda_max', type=float, default=float('inf'), help='max Lambda')
    parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
    parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete', 'Sigmoid', 'Tanh'])
    parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')
    args = parser.parse_args()

    
    # Learning schedules
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if args.epochs >= 110:
                # Train Wide-ResNet
                if t / args.epochs < 0.5:
                    return args.lr_max
                elif t / args.epochs < 0.75:
                    return args.lr_max / 10.
                elif t / args.epochs < (11 / 12):
                    return args.lr_max / 100.
                else:
                    return args.lr_max / 200.
            else:
                # Train ResNet
                if t / args.epochs < 0.3:
                    return args.lr_max
                elif t / args.epochs < 0.6:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                          [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t // (args.epochs // 10)) * (args.lr_max / 10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

    print(args)

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    main(args)

    # logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])
'''
    # Save the best checkpoint
    if test_pgd20_acc > best_acc:
        best_acc = test_pgd20_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            },filename='bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            })
'''
# logger_test.close()