# python main.py --num 3 --dataset multi --s real --t painting --gpu_id 0 --train 0 --seed 2022
# python main.py --num 1 --dataset office_home --s Clipart --t Art --gpu_id 0 --train 1 --seed 2021 --net vgg
import argparse
import os
import os.path as osp
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from models.SSDA_basenet import *
from models.SSDA_resnet import *
from copy import deepcopy
import contextlib
from data_pro.return_dataset import return_dataset, return_dataloader_by_UPS, TransformFixMatch
import scipy
import scipy.stats
from itertools import cycle
from soft_supconloss import SoftSupConLoss
import torchvision
import utils
from elr_loss import elr_loss
from torch.nn.parallel import DataParallel
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def train_source(args):
    source_loader,source_val_loader, _, _, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)
    netF = netF.cuda()
    netC = netC.cuda()
    param_group = []
    learning_rate = args.lr
    for k, v in netF.features.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate*10}]
    for k, v in netC.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate*10}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=0.5, patience=50,
                                                                verbose=True, min_lr=1e-6)
    acc_init = 0
    for epoch in (range(args.max_epoch)):
        netF.train()
        netC.train()
        total_losses,recon_losses,classifier_losses=[],[],[]
        iter_source = iter(source_loader)
        for _, (inputs_source, labels_source) in tqdm(enumerate(iter_source), leave=False):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            embeddings=netF(inputs_source)
            outputs_source = netC(embeddings)
            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source,
                                                                                   labels_source,T=1)
            total_loss=classifier_loss
            total_losses.append(total_loss.item())
            classifier_losses.append(classifier_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        netF.eval()
        netC.eval()
        scheduler.step(np.mean(total_losses))
        acc_s_tr, _ = cal_acc(source_loader, netF, netC)
        acc_s_te, _ = cal_acc(source_val_loader, netF, netC)
        log_str = 'Train on source, Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/{:.2f}%, total_loss: {:.6f}, classify loss: {:.6f},'.format(args.s+" to "+args.t, epoch + 1, args.max_epoch,
                                                                             acc_s_tr * 100, acc_s_te * 100,np.mean(total_losses),np.mean(classifier_losses))
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)
        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
            torch.save(best_netF, osp.join(args.output_dir, "source_F_val.pt"))
            torch.save(best_netC, osp.join(args.output_dir, "source_C_val.pt"))
    return netF, netC


def test_target(args):
    _, _, _, _, _, target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)
    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC = netC.cuda()
    netF = netF.cuda()
    netF.eval()
    netC.eval()
    acc, _ = cal_acc(target_loader_test, netF, netC)
    log_str = 'Test on target, Task: {}, Accuracy = {:.2f}%'.format(args.s+"2"+args.t, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)


def train_target(args):
    source_loader, source_val_loader, target_loader, target_loader_unl, \
    target_loader_val, target_loader_test, class_list = return_dataset(args)
    len_target_loader = len(target_loader.dataset)
    len_target_loader_unl = len(target_loader_unl.dataset)
    netF, netC, netD = get_model(args)
    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF = netF.cuda()
    netC = netC.cuda()
    param_group = []
    for k, v in netF.features.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr*10}]
    for k, v in netC.named_parameters():
        v.requires_grad = False
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=30,
                                                           verbose=True, min_lr=1e-6)
    supcon_loss = SoftSupConLoss(temperature=0.15)
    Elr_loss = elr_loss(len_target_loader_unl, args.class_num)
    easy_path, hard_path = data_split(args, netC, netF, target_loader_unl)
    dset_loaders = data_load_list(args, easy_path)
    proxy_loader_iter = iter(dset_loaders)
    dset_loaders_hard = data_load_list(args, hard_path)
    proxy_loader_iter_hard = iter(dset_loaders_hard)

    max_pred_acc = -1
    best_test_acc = -1
    best_F, bestC = None, None
    first_epoch_acc = -1
    psudo_acc = -1
    for epoch in (range(args.max_epoch)):
        if args.max_num_per_class > 0:
            active_target_loader, active_target_loader_unl, psudo_acc = return_dataloader_by_UPS(args,netF,netC,target_loader_unl)
            target_loader = active_target_loader
            target_loader_unl = active_target_loader_unl
        total_losses, l_classifier_losses, unl_classifier_losses, entropy_losses, labeled_entropy_losses, div_losses, m_losses= [], [],[],[],[],[],[]
        netF.eval()
        netC.eval()
        if epoch % 1 == 0:
            mem_label = label_propagation(target_loader_test, target_loader, netF, netC, args)
            mem_label = torch.from_numpy(mem_label)
        netF.train()
        netC.train()
        for step, ((labeled_target, label), (unlabeled_target, labels_target, idx)) in tqdm(enumerate(zip(cycle(target_loader), target_loader_unl)), desc='Training', leave=False):
            if unlabeled_target[0].size(0) == 1:
                continue
            len_label = labeled_target.size(0)
            len_unlabel = unlabeled_target[0].size(0)

            unlabeled_target_strong = unlabeled_target[1]
            unlabeled_target = unlabeled_target[0]

            try:
                inputs_proxy, labels_proxy = proxy_loader_iter.next()
            except:
                proxy_loader_iter = iter(dset_loaders)
                inputs_proxy, labels_proxy = proxy_loader_iter.next()
            inputs_proxy_strong = inputs_proxy[1]
            inputs_proxy = inputs_proxy[0]
            try:
                inputs_proxy_hard, labels_proxy_hard = proxy_loader_iter_hard.next()
            except:
                proxy_loader_iter_hard = iter(dset_loaders_hard)
                inputs_proxy_hard, labels_proxy_hard = proxy_loader_iter_hard.next()
            inputs_proxy_hard_strong = inputs_proxy_hard[1]
            inputs_proxy_hard = inputs_proxy_hard[0]

            inputs = torch.cat((labeled_target, unlabeled_target, unlabeled_target_strong), dim=0).cuda()
            target_features = netF(inputs)
            target_out = netC(target_features, reverse=False)
            labels_target = labels_target.cuda()
            unlabeled_target_out_strong = target_out[len_label+len_unlabel:]
            unlabeled_target_out = target_out[len_label:len_label+len_unlabel]
            labeled_target_pred = target_out[0:len_label]

            classifier_loss = 0
            im_loss = 0
            con_loss = 0

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0)(labeled_target_pred, label)
            l_classifier_losses.append(classifier_loss.item())

            # ELR loss
            pred_label = mem_label[idx]
            unl_loss = args.unl_w * Elr_loss(idx, unlabeled_target_out, pred_label.cuda())
            classifier_loss += unl_loss
            unl_classifier_losses.append(unl_loss.item())

            # Prob contra loss
            with torch.no_grad():
                select_matrix = contrast_left_out_with_mask(torch.ones(len_unlabel).cuda())# no masks
            softmax_out = nn.Softmax(dim=1)(unlabeled_target_out)
            softmax_out_strong = nn.Softmax(dim=1)(unlabeled_target_out_strong)
            max_probs, _ = torch.max(softmax_out, dim=1)
            con_loss += 0.1 * supcon_loss(torch.stack((softmax_out, softmax_out_strong), dim=1), max_probs, pred_label, select_matrix=select_matrix)
            m_losses.append(con_loss.item())

            # Mixup loss
            lam = np.random.beta(0.75, 0.75)
            lam = max(lam, 1 - lam)
            labeled_target, unlabeled_target, unlabeled_target_strong, inputs_proxy, inputs_proxy_strong, inputs_proxy_hard, inputs_proxy_hard_strong = \
            labeled_target.cuda(), unlabeled_target.cuda(), unlabeled_target_strong.cuda(), inputs_proxy.cuda(), inputs_proxy_hard.cuda(), inputs_proxy_strong.cuda(), inputs_proxy_hard_strong.cuda()
            targets_proxy = torch.zeros(len_unlabel, args.class_num).scatter_(1, labels_proxy.view(-1,1).long(), 1).cuda()
            target_out_proxy = netC(netF(torch.cat([inputs_proxy_hard, inputs_proxy_hard_strong], dim=0)))
            unlabeled_target_out_proxy_hard, unlabeled_target_out_proxy_hard_strong = target_out_proxy.chunk(2)
            with torch.no_grad():
                p_hard = (nn.Softmax(dim=1)(unlabeled_target_out_proxy_hard)+nn.Softmax(dim=1)(unlabeled_target_out_proxy_hard_strong))/2
                pt_hard = p_hard ** (1 / 0.5)
                targets_proxy_hard = pt_hard / pt_hard.sum(dim=1, keepdim=True)
                targets_proxy_hard = targets_proxy_hard.detach().cuda()
            all_inputs_s = torch.cat([inputs_proxy, inputs_proxy_strong], dim=0)
            all_inputs_t = torch.cat([inputs_proxy_hard, inputs_proxy_hard_strong], dim=0)
            all_targets_s = torch.cat([targets_proxy, targets_proxy], dim=0)
            all_targets_t = torch.cat([targets_proxy_hard, targets_proxy_hard], dim=0)
            idx = torch.randperm(all_inputs_s.size(0))
            input_a, input_b = all_inputs_s, all_inputs_t[idx]
            target_a, target_b = all_targets_s, all_targets_t[idx]
            mixed_input = lam * input_a + (1 - lam) * input_b
            mixed_target = lam * target_a + (1 - lam) * target_b
            mixed_inputs = interleave(mixed_input, len_unlabel)
            mixed_logits = netC(netF(mixed_inputs))
            mixed_logits = de_interleave(mixed_logits, len_unlabel)
            mixed_probs = torch.softmax(mixed_logits, dim=1)
            w = 60 * exp_rampup(epoch + step / len_target_loader_unl, args.max_epoch)
            con_loss += w * torch.mean((mixed_probs - mixed_target) ** 2)
            m_losses.append(con_loss.item())

            softmax_out = nn.Softmax(dim=1)(unlabeled_target_out)
            un_labeled_entropy = torch.mean(Entropy(softmax_out))
            im_loss += args.unlent * un_labeled_entropy
            entropy_losses.append(un_labeled_entropy.item())

            msoftmax = softmax_out.mean(dim=0)
            tmp = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
            div_losses.append(tmp.item())

            im_loss -= args.div_w * tmp
            total_loss = args.im * im_loss + args.par * classifier_loss + con_loss
            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        netF.eval()
        netC.eval()
        acc, _ = cal_acc(target_loader_test, netF, netC, tsne=True, epoch=epoch)
        acc_val, _ = cal_acc(target_loader_val, netF, netC, tsne=False, epoch=epoch)
        scheduler.step(np.mean(total_losses))
        log_str = 'Training the model in target: {}, epoch:{}/{}; acc_test = {:.2f}%, acc_val = {:.2f}%, total_loss: {:.2f}, L_cls_loss: {:.2f}, UNL_cls_loss: {:.2f}, ' \
                  'ent_loss: {:.2f}, div_loss: {:.2f}, contra_loss: {:.2f}, pseudo_label_acc: {:.2f}%'.format(
                    args.s+" to "+args.t, epoch+1, args.max_epoch, acc*100, acc_val*100,
                    np.mean(total_losses), np.mean(l_classifier_losses), np.mean(unl_classifier_losses), np.mean(entropy_losses),
                    np.mean(div_losses), np.mean(m_losses), psudo_acc)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)
        if max_pred_acc < acc_val:
            best_F = deepcopy(netF)
            best_C = deepcopy(netC)
            max_pred_acc = acc_val
        if best_test_acc < acc:
            best_test_acc = acc
    acc, _ = cal_acc(target_loader_test, best_F, best_C)
    log_str = "Test acc: {:.4f} ".format(acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return netF, netC


@torch.no_grad()
def data_split(args, netC, netF, target_loader_unl):
    NUMS = 15  # easy samples of each class
    NUMS_hard = 15  # hard samples of each class
    easy_path, hard_path, easy_idx, hard_idx = [], [], [], []
    netC.eval()
    netF.eval()
    basepath = os.path.join("/home/user/huangxinyang/SSDA/code/data/txt/%s"%args.dataset,
                    'unlabeled_target_images_' + args.t + '_%d.txt' % (args.num))
    targetpath = os.path.join("/home/user/huangxinyang/SSDA/code/data/%s"%args.dataset)
    img, label = utils.make_dataset_fromlist(open(basepath).readlines())
    for name, param in netC.named_parameters():
        if('fc.weight' in name):
            prototype = param
    _, features_bank = get_embedding(args, target_loader_unl, netC, netF)
    features_bank = F.normalize(features_bank) # len * 256
    prototype = F.normalize(prototype) # cls * 256
    dists = prototype.mm(features_bank.t())  # cls * len
    sort_idxs = torch.argsort(dists, dim=1, descending=True) # cls * len
    sort_idxs_hard = torch.argsort(dists, dim=1) # cls * len
    fault = 0.
    print('Starting generating proxy...')
    args.out_file.write('Starting generating proxy...')
    args.out_file.flush()
    for i in tqdm(range(args.class_num)):
        s_idx = 0
        for _ in range(NUMS):
            idx = sort_idxs[i, s_idx]
            while idx in easy_idx:
                s_idx += 1
                idx = sort_idxs[i, s_idx]
            assert idx not in easy_idx
            easy_idx.append(idx)
            easy_path.append((targetpath + "/" + img[idx], i))
            if not label[idx] == i:
                fault += 1
            s_idx += 1
    for i in tqdm(range(args.class_num)):
        s_idx = 0
        for _ in range(NUMS_hard):
            idx = sort_idxs_hard[i, s_idx]
            while idx in hard_idx:
                s_idx += 1
                idx = sort_idxs_hard[i, s_idx]
            assert idx not in hard_idx
            hard_idx.append(idx)
            hard_path.append((targetpath + "/" + img[idx], i))
            s_idx += 1
    acc = 1 - fault / (args.class_num * NUMS)
    print('Splited data list label Acc: {}'.format(acc))
    args.out_file.write('Splited data list label Acc: {}'.format(acc) + '\n')
    args.out_file.flush()
    return easy_path, hard_path


@torch.no_grad()
def get_embedding(args, tgt_loader, netC, netF, cat_data=False, aug=False):
    netC.eval()
    netF.eval()
    pred_bank = torch.zeros([len(tgt_loader.dataset), args.class_num]).cuda()
    emb_bank = torch.zeros([len(tgt_loader.dataset), args.bottleneck]).cuda()
    for batch_idx, (data, target, idx) in enumerate(tgt_loader):
        data, target = data.cuda(), target.cuda()
        fea = netF(data)
        out = netC(fea)
        emb_bank[idx] = fea
        pred_bank[idx] = out
    return pred_bank, emb_bank


def exp_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(np.exp(2*current / rampup_length)-0.99, 0.0, 1.0)
        return float(current)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def data_load_list(args, p_list):
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((crop_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_transform_fix = torchvision.transforms.Compose(
        [TransformFixMatch(crop_size=crop_size)])
    source_set = utils.ObjectImage_list(p_list, train_transform_fix)
    dset_loaders = torch.utils.data.DataLoader(source_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True)
    return dset_loaders


def de_interleave(x, batch_size):
    s = list(x.shape)
    return x.reshape([batch_size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def interleave(x, batch_size):
    s = list(x.shape)
    return x.reshape([-1, batch_size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def contrast_left_out_with_mask(mask):
    contrast_mask = mask
    contrast_mask2 = torch.clone(contrast_mask)
    contrast_mask2[contrast_mask == 0] = -1
    select_elements = torch.eq(contrast_mask2.reshape([-1, 1]), contrast_mask.reshape([-1, 1]).T).float()
    select_elements += torch.eye(contrast_mask.shape[0]).cuda()
    select_elements[select_elements > 1] = 1
    select_matrix = torch.ones(contrast_mask.shape[0]).cuda() * select_elements
    return select_matrix


def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "=========================================="
    print(s)
    return s


def inferance(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_fea,all_output,all_label


def label_propagation(loader, labeled_loader, netF, netC, args):
    all_fea, all_output, all_label = inferance(loader, netF, netC, args)
    K = all_output.size(1)
    all_fea_labeled, all_output_labeled, all_label_labeled = inferance(labeled_loader, netF, netC, args)
    max_iter = 40
    alpha = 0.90
    if args.dataset =="multi":
        k = 10 # deafult 10
    elif args.dataset =="office_home":
        k = 7  # deafult 10
    else:
        k = 5 # deafult 10
    log_str = 'Processing label propagation...'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

    labels = np.asarray(torch.cat((all_label_labeled, all_label), 0).numpy())
    labeled_idx = np.asarray(range(len(labels))[0:len(all_label_labeled)])
    unlabeled_idx = np.asarray(range(len(labels))[len(all_label_labeled):])
    with torch.no_grad():
        output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        Fea = torch.cat((all_fea_labeled, all_fea), 0)
        N=Fea.shape[0]
        X = F.normalize(Fea, dim=1)
        simlarity_matrix = X.matmul(X.transpose(0, 1))
        D, I = torch.topk(simlarity_matrix, k + 1)
        D = D.cpu().numpy()
        I = I.cpu().numpy()
        # Create the graph
        D = D[:, 1:] ** 4
        I = I[:, 1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (k, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T
        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D
        # Initiliaze the y vector for each class and apply label propagation
        Z = np.zeros((N, K))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(K):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            if not cur_idx.shape[0]==0:
                y[cur_idx] = 1.0 / cur_idx.shape[0]
            else:
                y[cur_idx] = 1.0

            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f
        # Handle numberical errors
        Z[Z < 0] = 0
        # Compute the weight for each instance based on the entropy 
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        p_labels = np.argmax(probs_l1, 1)
        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels[unlabeled_idx] == labels[unlabeled_idx])
        acc = correct_idx.mean()
        p_labels[labeled_idx] = labels[labeled_idx]
    log_str = 'LP Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
    return p_labels[len(all_label_labeled):].astype('int') 


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets,T=1.0):
        log_probs = self.logsoftmax(inputs/T)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def cal_acc(loader, netF, netC, tsne=True, epoch=0):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = netC(netF(inputs))

            labels = labels.cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def get_model(args):
    netF, netC, netD = None, None, None
    if args.net == 'resnet34':
        netF = resnet34(args=args)
        inc = args.bottleneck
        netC = Predictor(num_class=args.class_num, inc=inc, norm_feature=args.norm_feature, temp=args.temp)
    elif args.net == "alexnet":
        inc = args.bottleneck
        netF = AlexNetBase(bootleneck_dim=inc)
        netC = Predictor(num_class=args.class_num, inc=inc, norm_feature=args.norm_feature, temp=args.temp)
    elif args.net == "vgg":
        inc = args.bottleneck
        netF = VGGBase(bootleneck_dim=inc)
        netC = Predictor(num_class=args.class_num, inc=inc, norm_feature=args.norm_feature, temp=args.temp)
    else:
        raise ValueError('Model cannot be recognized.')
    return  netF, netC, netD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=int, default=1, help="device id to run")
    parser.add_argument('--net', type=str, default="resnet34", choices=['vgg', "alexnet", "resnet34"])
    parser.add_argument('--s', type=str, default="webcam", help="source office_home :Art Clipart Product Real_World")
    parser.add_argument('--t', type=str, default="amazon", help="target  Art Clipart Product Real_World")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch ")
    parser.add_argument('--num', type=int, default=1, help="labeled_data per class. 1: 1-shot, 3: 3-shot")
    parser.add_argument('--train', type=int, default=1, help="if to train")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--class_num', type=int, default=65, help="batch_size",choices=[65,31,126])
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dataset', type=str, default='Office-31', choices=['office_home', 'multi', 'Office-31'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--norm_feature', type=int, default=1, help="random seed")
    parser.add_argument('--par', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--im', type=float, default=1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='SSDA')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--unlent', type=float, default=1)
    parser.add_argument('--unl_w', type=float, default=0.5)
    parser.add_argument('--vat_w', type=float, default=0.1)
    parser.add_argument('--div_w', type=float, default=1)
    parser.add_argument('--max_num_per_class', type=int, default=1, help="max-number per class to select, the MNPC in paper")
    parser.add_argument('--uda', type=int, default=0, help="if to perform unsurpervised DA")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import warnings
    warnings.filterwarnings("ignore")
    current_folder = "./log"

    if args.dataset== "Office-31":
        args.class_num=31
        args.batch_size=64
        args.unl_w=0.5
    elif args.dataset== "office_home":
        args.class_num=65
        args.batch_size=16
        args.unl_w=0.3
    elif args.dataset== "multi":
        args.class_num=126
        # args.batch_size=64
        args.batch_size=48
        args.unl_w=0.3
    else:
        print("We do not have the dataset", args.dataset)

    args.output_dir = osp.join(current_folder, args.output, 'seed' + str(args.seed), args.dataset,args.s+"_lr"+str(args.lr))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.train == 1:
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
    else:
        args.out_file = open(osp.join(args.output_dir, 'Target'+args.s+"to"+args.t+"_lr"+str(args.lr)
                                      +"_unl_ent"+str(args.unlent)+ "_unl_w"+str(args.unl_w)+
                                      "_vat_w"+str(args.vat_w)+ "_div_w"+str(args.div_w)+
                                      "_MNPC"+str(args.max_num_per_class)+"_num"+str(args.num)+'.txt'), 'w')
        test_target(args)
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)