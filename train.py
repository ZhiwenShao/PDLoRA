import argparse
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as util_data
import itertools

import network
import pre_process as prep
from util_new import *
from data_list import ImageList_au

import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random


from timm.models import create_model
from optim_factory import create_optimizer

import utils_from_rest
from utils_from_rest import NativeScalerWithGradNormCount as NativeScaler

import warnings

warnings.filterwarnings('ignore')

def BCEWithLogitsLoss_PNWeight(input, target, p_n_weight, weight=None, size_average=True, reduce=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
                is ``False``, returns a loss per input/target element instead and ignores
                :attr:`size_average`. Default: ``True``

    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    # make all the negative value to positive and positive values to 0
    max_val = (-input).clamp(min=0)
    loss = (p_n_weight - 1) * target * (1 + (- input).exp()).log() + \
        input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = torch.relu(loss)
    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def attention_KLDiv_loss(input, target, size_average=True, reduce=True):
    classify_loss = nn.KLDivLoss(reduction='batchmean', size_average=size_average, reduce=reduce)

    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def AttentionLoss(x):
        eps = sys.float_info.epsilon
        # num_head = len(x)
        num_head = x.size(2)
        au_num = x.size(1)
        
        if num_head > 1:
            for k in range(au_num):
                loss = 0
                cnt = 0
                for i in range(num_head-1):
                    for j in range(i+1, num_head):
                        mse = F.mse_loss(x[:,k,i,:,:,:], x[:,k,j,:,:,:])
                        cnt = cnt+1
                        loss = loss+mse
                loss = cnt/(loss + eps)
                loss = torch.unsqueeze(loss, 0)
                if k == 0:
                    all_loss = loss
                else:
                    all_loss = torch.cat((all_loss, loss), 0)
            return all_loss.mean()
        else:
            return 0


class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2


def set_random_seed(SEED=4):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config):
    # fix the seed for reproducibility
    set_random_seed(config.seed)
    
    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt', dtype=float))
    au_p_n_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_p_n_weight.txt', dtype=float))
    if use_gpu:
        au_weight = au_weight.float().cuda()
        au_p_n_weight = au_p_n_weight.float().cuda()
    else:
        au_weight = au_weight.float()
        au_p_n_weight = au_p_n_weight.float()

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = ImageList_au(crop_size=config.crop_size, path=config.train_path_prefix,
                                        transform=prep.image_train_1(crop_size=config.crop_size, re_size=config.width * 14))

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)

    dsets['test'] = ImageList_au(config.test_path_prefix, phase='test', 
                                      transform=prep.image_test_1(crop_size=config.crop_size, re_size=config.width * 14))

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    ## set network modules
    backbone = network.network_dict[config.backbone](
        lora_depth = config.lora_depth,
        r = config.rank,
        rand_rank = config.rand_rank,
        drop_rate = config.drop_rate
    )

    au_net = network.network_dict[config.au_net](
        au_num=config.au_num,
        num_head=config.num_head, 
        channel=config.channel, 
        width=config.width
    )

    if config.pretrain:
        print('use pretraining')
        
        backbone.load_state_dict(torch.load(
            config.write_path_prefix + config.pretrain_path + '/backbone_' + str(
                config.pretrain_epoch) + '.pth'))#strict=False
        au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.pretrain_path + '/au_net_' + str(
                config.pretrain_epoch) + '.pth'))

        for i in range(config.backbone_blocks-config.lora_depth, config.backbone_blocks):
            print('setting block ', i)
            qkv_lora_A_part1 = backbone.blocks[i].attn.qkv.lora_A[0:int(backbone.blocks[i].attn.qkv.lora_A.shape[0]/2),:]
            qkv_lora_A_part2 = backbone.blocks[i].attn.qkv.lora_A[int(backbone.blocks[i].attn.qkv.lora_A.shape[0]/2):backbone.blocks[i].attn.qkv.lora_A.shape[0],:]
            lora_A_part1, _, _ = torch.svd_lowrank(qkv_lora_A_part1.T, config.rank-config.stride)
            lora_A_part2, _, _ = torch.svd_lowrank(qkv_lora_A_part2.T, config.rank-config.stride)
            backbone.blocks[i].attn.qkv.lora_A = nn.Parameter(torch.cat((lora_A_part1.T, lora_A_part2.T), dim=0))
            lora_B, _, _ = torch.svd_lowrank(backbone.blocks[i].attn.qkv.lora_B, config.rank-config.stride)
            backbone.blocks[i].attn.qkv.lora_B = nn.Parameter(lora_B)
            print(backbone.blocks[i].attn.qkv.lora_A.shape, backbone.blocks[i].attn.qkv.lora_B.shape)

    if config.start_epoch > 0:
        print('resuming model from epoch %d' %(config.start_epoch))
        
        backbone.load_state_dict(torch.load(
            config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/backbone_' + str(
                config.start_epoch) + '.pth'))#strict=False
        au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/au_net_' + str(
                config.start_epoch) + '.pth'))         

    if use_gpu:
        backbone = backbone.cuda()
        au_net = au_net.cuda()

    net = EnsembleModel(backbone, au_net)

    num_training_steps_per_epoch = len(dset_loaders['train'])
    config.lr = config.lr * config.train_batch_size / 256

    ## set optimizer
    optimizer = create_optimizer(config, net.parameters())
    loss_scaler = NativeScaler()
    
    print("Use Cosine LR scheduler")
    lr_schedule_values = utils_from_rest.cosine_scheduler(
        config.lr, config.min_lr, config.epochs, num_training_steps_per_epoch,
        warmup_epochs=config.warmup_epochs, warmup_steps=config.warmup_steps,
    )

    if config.weight_decay_end is None:
        config.weight_decay_end = config.weight_decay
    wd_schedule_values = utils_from_rest.cosine_scheduler(
        config.weight_decay, config.weight_decay_end, config.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if not os.path.exists(config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name):
        os.makedirs(config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name):
        os.makedirs(config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name)

    res_file = open(
        config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '.txt', 'w')
    res_file2 = open(
        config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '_details.txt', 'w')

    ## train
    count = 0

    for epoch in range(config.start_epoch, config.epochs + 1):
        if epoch > config.start_epoch:
            print('taking snapshot ...')
            torch.save(net.modelA.state_dict(),
                        config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/backbone_' + str(
                            epoch) + '.pth')
            torch.save(net.modelB.state_dict(),
                        config.write_path_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/au_net_' + str(
                            epoch) + '.pth')            
            
        # eval in the train
        if epoch > config.start_epoch:
            print('testing ...')
            net.train(False)

            f1score_arr, acc_arr = AU_detection_eval_both(dset_loaders['test'], net.modelA, net.modelB, use_gpu=use_gpu)
            print('epoch =%d, f1 score mean=%f, accuracy mean=%f' %
                  (epoch, f1score_arr.mean(), acc_arr.mean()))
            print('%d\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean()), file=res_file)
            print(f1score_arr, acc_arr, file=res_file2)
            res_file.close()
            res_file2.close()
            res_file = open(
                config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/AU_pred_' + str(
                    config.start_epoch) + '.txt', 'a')
            res_file2 = open(
                config.write_res_prefix + config.backbone + '_' + config.au_net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '_details.txt', 'a')   

            net.train(True)

        if epoch >= config.epochs:
            break

        start_steps = epoch * num_training_steps_per_epoch
        optimizer.zero_grad()
        for i, batch in enumerate(dset_loaders['train']):

            step = i // config.update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None or wd_schedule_values is not None and i % config.update_freq == 0:
                for j, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            if i % config.display == 0 and count > 0:
                if config.num_head > 1:
                    print('[epoch = %d][iter = %d][total_loss = %f][loss_au = %f][loss_attention = %f]' % (epoch, i,
                                                                                                       total_loss.data.cpu().numpy(),
                                                                                                       loss_au.data.cpu().numpy(),
                                                                                                       loss_attention.data.cpu().numpy()))
                else:
                    print('[epoch = %d][iter = %d][total_loss = %f][loss_au = %f]' % (epoch, i,
                                                                                                       total_loss.data.cpu().numpy(),
                                                                                                       loss_au.data.cpu().numpy()))
                print('learning rate = %f' % (optimizer.param_groups[0]['lr']))
                print('weight decay = %f' % (optimizer.param_groups[0]['weight_decay']))
                print('the number of training iterations is %d' % (count))

            img, au = batch

            if use_gpu:
                img, au = img.cuda(), au.float().cuda()
            else:
                au = au.float()

            if config.use_amp:
                with torch.cuda.amp.autocast():                    
                    feat = net.modelA(img)
                    aus_attention, aus_output = net.modelB(feat)
                    
                    if config.num_head > 1:
                        loss_attention = AttentionLoss(aus_attention)  
                        loss_au = BCEWithLogitsLoss_PNWeight(aus_output, au, au_p_n_weight, au_weight)
                        total_loss = config.lambda_au * loss_au + config.lambda_attention * loss_attention
                    else:
                        loss_au = BCEWithLogitsLoss_PNWeight(aus_output, au, au_p_n_weight, au_weight)
                        total_loss = config.lambda_au * loss_au

                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    total_loss /= config.update_freq
                    grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.clip_grad,
                                            parameters=net.parameters(), create_graph=is_second_order,
                                            update_grad=(i + 1) % config.update_freq == 0)
                    if (i + 1) % config.update_freq == 0:
                        optimizer.zero_grad()

            torch.cuda.synchronize()
            count = count + 1


    res_file.close()
    res_file2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=182, help='crop size for images')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--pretrain', type=str2bool, default=False)
    parser.add_argument('--rand_rank', type=str2bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=8, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--backbone', type=str, default='dinov2_lora')
    parser.add_argument('--au_net', type=str, default='au_ddam')
    parser.add_argument('--run_name', type=str, default='v1')
    parser.add_argument('--dataset_name', type=str, default='BP4D')

    # Training configuration.
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_attention', type=float, default=0.05, help='weight for attention loss')
    parser.add_argument('--display', type=int, default=100, help='iteration gaps for displaying')
    parser.add_argument('--rank_gap', type=int, default=100)
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")     
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop_rate for LoRA')                                   

    # Directories.
    parser.add_argument('--pretrain_path', type=str, default='dinov2_lora_au_ddam/v1/')
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/BP4D_combine_1_2')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/BP4D_part3')

    #----------------------------From ResT---------------------------------
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--pretrain_epoch', default=10, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--lora_depth', type=int, default=13)
    parser.add_argument('--backbone_blocks', type=int, default=40)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--channel', type=int, default=1536)
    parser.add_argument('--width', type=int, default=13)

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-2, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--inter_epoch', type=int, default=0)                    

    parser.add_argument('--seed', default=4, type=int)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)