import gc
import sys
import pickle

import torch
import scipy.linalg as scilin

import models
from models.res_adapt import ResNet18_adapt
from utils import *
from args import parse_eval_args
from datasets import make_dataset

import torch.nn.functional as F

MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)

SVHN_TRAIN_SAMPLES = (4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659)
SVHN_TEST_SAMPLES = (1744, 5099, 4149, 2882, 2523, 2384, 1977, 2019, 1660, 1595)
FASHION_MNIST_TRAIN_SAMPLES = 10 * (6000,)
FASHION_MNIST_TEST_SAMPLES = 10 * (1000,)


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, dataloader, isTrain=True):
    mu_G = 0
    mu_c_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] += features[b, :]

        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    if args.dataset == 'mnist':
        if isTrain:
            mu_G /= sum(MNIST_TRAIN_SAMPLES)
            for i in range(len(MNIST_TRAIN_SAMPLES)):
                mu_c_dict[i] /= MNIST_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(MNIST_TEST_SAMPLES)
            for i in range(len(MNIST_TEST_SAMPLES)):
                mu_c_dict[i] /= MNIST_TEST_SAMPLES[i]
    elif args.dataset == 'fashionmnist':
        if isTrain:
            mu_G /= sum(FASHION_MNIST_TRAIN_SAMPLES)
            for i in range(len(FASHION_MNIST_TRAIN_SAMPLES)):
                mu_c_dict[i] /= FASHION_MNIST_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(FASHION_MNIST_TEST_SAMPLES)
            for i in range(len(FASHION_MNIST_TEST_SAMPLES)):
                mu_c_dict[i] /= FASHION_MNIST_TEST_SAMPLES[i]
    elif args.dataset == 'svhn':
        if isTrain:
            mu_G /= sum(SVHN_TRAIN_SAMPLES)
            for i in range(len(SVHN_TRAIN_SAMPLES)):
                mu_c_dict[i] /= SVHN_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(SVHN_TEST_SAMPLES)
            for i in range(len(SVHN_TEST_SAMPLES)):
                mu_c_dict[i] /= SVHN_TEST_SAMPLES[i]
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            mu_G /= sum(CIFAR10_TRAIN_SAMPLES)
            for i in range(len(CIFAR10_TRAIN_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(CIFAR10_TEST_SAMPLES)
            for i in range(len(CIFAR10_TEST_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TEST_SAMPLES[i]

    return mu_G, mu_c_dict, top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, isTrain=True):
    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if args.dataset == 'mnist':
        if isTrain:
            Sigma_W /= sum(MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(MNIST_TEST_SAMPLES)
    elif args.dataset == "fashionmnist":
        if isTrain:
            Sigma_W /= sum(FASHION_MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(FASHION_MNIST_TEST_SAMPLES)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(CIFAR10_TEST_SAMPLES)
    elif args.dataset == "svhn":
        if isTrain:
            Sigma_W /= sum(SVHN_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(SVHN_TEST_SAMPLES)

    return Sigma_W.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh - b, p='fro')
    return res_b.detach().cpu().numpy().item()


def compute_ECE(args, model, dataloader, fc_features, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # TODO: Compute expected calibration error
    logits_list = []
    labels_list = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        logits_list.append(F.softmax(outputs[0].data, dim=-1))
        labels_list.append(targets.data)

    # Create tensors
    logits_list = torch.cat(logits_list).to(args.device)
    labels_list = torch.cat(labels_list).to(args.device)

    confidences, predictions = torch.max(logits_list, 1)
    accuracies = predictions.eq(labels_list)

    ece = torch.zeros(1, device=args.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    fc_features.clear()

    return ece.detach().cpu().numpy()  # .item()


def main():
    args = parse_eval_args()

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)

    if args.model == "MLP":
        model = models.__dict__[args.model](hidden=args.width, depth=args.depth, fc_bias=args.bias,
                                            num_classes=num_classes).to(device)
    elif args.model == "ResNet18_adapt":
        model = ResNet18_adapt(width=args.width, num_classes=num_classes, fc_bias=args.bias).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                            fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'W': [],
        'b': [],
        'H': [],
        'mu_G_train': [],
        # 'mu_G_test': [],
        'train_acc1': [],
        'train_acc5': [],
        'test_acc1': [],
        'test_acc5': [],
        'ece_metric_train': [],
        'ece_metric_test': []
    }

    logfile = open('%s/test_log.txt' % (args.load_path), 'w')
    for i in range(args.epochs):

        model.load_state_dict(torch.load(args.load_path + 'epoch_' + str(i + 1).zfill(3) + '.pth'))
        model.eval()

        for n, p in model.named_parameters():
            if 'fc.weight' in n:
                W = p
            if 'fc.bias' in n:
                b = p

        mu_G_train, mu_c_dict_train, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader,
                                                                           isTrain=True)
        mu_G_test, mu_c_dict_test, test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader,
                                                                       isTrain=False)

        Sigma_W = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, trainloader, isTrain=True)
        # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)
        WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
        if args.bias:
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)
        else:
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, torch.zeros((W.shape[0],)))

        ece_metric_train = compute_ECE(args, model, trainloader, fc_features)
        ece_metric_test = compute_ECE(args, model, testloader, fc_features)

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric)
        info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

        info_dict['ece_metric_train'].append(ece_metric_train)
        info_dict['ece_metric_test'].append(ece_metric_test)

        info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())
        info_dict['H'].append(H.detach().cpu().numpy())

        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        # info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)

        print_and_save(
            '[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f | train ECE: %.4f | test ECE: %.4f ' %
            (i + 1, train_acc1, train_acc5, test_acc1, test_acc5, ece_metric_train, ece_metric_test), logfile)

    with open(args.load_path + 'info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)


def add_ece():
    args = parse_eval_args()

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)

    if args.model == "MLP":
        model = models.__dict__[args.model](hidden=args.width, depth=args.depth, fc_bias=args.bias,
                                            num_classes=num_classes).to(device)
    elif args.model == "ResNet18_adapt":
        model = ResNet18_adapt(width=args.width, num_classes=num_classes, fc_bias=args.bias).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                            fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    print("Opening file " + args.load_path + 'info.pkl')
    with open(args.load_path + 'info.pkl', 'rb') as f:
        info_dict = pickle.load(f)

    info_dict["ece_metric_train"] = []
    info_dict["ece_metric_test"] = []

    logfile = open('%s/test_log2.txt' % (args.load_path), 'w')
    for i in range(args.epochs):
        model.load_state_dict(torch.load(args.load_path + 'epoch_' + str(i + 1).zfill(3) + '.pth'))
        model.eval()

        # mu_G_train, mu_c_dict_train, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader,
        #                                                                    isTrain=True)
        # mu_G_test, mu_c_dict_test, test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader,
        #                                                                isTrain=False)
        #
        # Sigma_W = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, trainloader, isTrain=True)
        # # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        # Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)
        #
        # collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        # ETF_metric = compute_ETF(W)
        # WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
        # if args.bias:
        #     Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)
        # else:
        #     Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, torch.zeros((W.shape[0],)))

        ece_metric_train = compute_ECE(args, model, trainloader, fc_features)
        ece_metric_test = compute_ECE(args, model, testloader, fc_features)

        torch.cuda.empty_cache()

        # info_dict['collapse_metric'].append(collapse_metric)
        # info_dict['ETF_metric'].append(ETF_metric)
        # info_dict['WH_relation_metric'].append(WH_relation_metric)
        # info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

        info_dict['ece_metric_train'].append(ece_metric_train)
        info_dict['ece_metric_test'].append(ece_metric_test)

        # info_dict['W'].append((W.detach().cpu().numpy()))
        # if args.bias:
        #     info_dict['b'].append(b.detach().cpu().numpy())
        # info_dict['H'].append(H.detach().cpu().numpy())
        #
        # info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        # # info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())
        #
        # info_dict['train_acc1'].append(train_acc1)
        # info_dict['train_acc5'].append(train_acc5)
        # info_dict['test_acc1'].append(test_acc1)
        # info_dict['test_acc5'].append(test_acc5)

        print_and_save(
            '[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f | train ECE: %.4f | test ECE: %.4f ' %
            (i + 1, info_dict['train_acc1'][i], info_dict['train_acc5'][i], info_dict['test_acc1'][i],
             info_dict['test_acc5'][i], ece_metric_train, ece_metric_test), logfile)

    # TODO: Uncomment me
    with open(args.load_path + 'info2.pkl', 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    main()
    # add_ece()
