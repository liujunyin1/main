# -*- coding: utf-8 -*-
# 该文件为“SynFoC：基础模型(MedSAM) + 常规模型(U-Net)的协同训练”实验脚本的详尽注释版
# 注释重点：
# 1) 对应论文的关键思想标注（SMC 自互信、CDCR 共识-发散一致性正则化、拷贝粘贴中间样本、Mean Teacher/EMA 等）【论文 §3，图2、图4，式(1)(2)(3)(4)(5)(6)(7)(9)(10)】:contentReference[oaicite:2]{index=2}
# 2) 解释训练/验证数据流、伪标签集成、阈值与遮罩如何落地到张量运算
# 3) 解释混合精度、优化器/学习率、EMA、日志指标（Dice/Jaccard/95HD/ASD）等工程细节
# 原始代码来源并保持功能不变，仅增补中文注释以便复现与研究。:contentReference[oaicite:3]{index=3}
# 修复loss为0的bug，完善test函数中loss计算。修复train函数中'--eval'变量名错误。修复model_name变量未定义错误。
# 增加了定义输出路径
import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from networks.unet_model import UNet                  # 常规模型：U-Net（论文统一对齐传统模型）:contentReference[oaicite:4]{index=4}
from networks.wrn import build_WideResNet            # （未使用）WideResNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation, BUSISegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util       # Dice 损失/指标，sigmoid ramp-up 等
from torch.cuda.amp import autocast, GradScaler      # 混合精度训练
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math
from medpy.metric import binary                      # 经典医学分割几何指标：Dice、Jaccard、HD95、ASD
from segment_anything import sam_model_registry       # 基础模型：SAM/MedSAM 注册与构建（论文将 MedSAM 作为 Foundation Model）:contentReference[oaicite:5]{index=5}
from importlib import import_module                   # 动态导入 LoRA 模块（参数高效微调）
from scipy.ndimage import zoom
import cv2
from itertools import chain
from skimage.measure import label

# -----------------------------
# 参数解析：覆盖数据、模型、优化、训练策略（含 SMC/CDCR 所需阈值/超参）
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI']) # 数据集选择
parser.add_argument("--save_name", type=str, default="", help="experiment_name") # 实验名称（用于日志/模型保存目录）
parser.add_argument("--overwrite", action='store_true') # 是否覆盖已有实验目录
parser.add_argument("--model", type=str, default="MedSAM", help="model_name") # 模型名称（仅日志使用）
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train") # 最大训练迭代次数
parser.add_argument('--num_eval_iter', type=int, default=500) # 每隔多少迭代评估一次（即每个 epoch 的迭代数）
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training") # 是否使用确定性训练，以保证实验结果的可复现性
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate") # 基础学习率
parser.add_argument("--seed", type=int, default=1337, help="random seed") # 随机种子
parser.add_argument("--gpu", type=str, default='0') # 使用的 GPU 设备 ID
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)
# ↑ 论文式(7)中用于高置信区域过滤的 τ=0.95（高置信伪标签遮罩 Wc^EN），这里以参数形式提供。:contentReference[oaicite:6]{index=6}

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not') # 是否使用混合精度训练（加速与节省显存）

# 修Bug参数，原本缺失
parser.add_argument('--eval', action='store_true', # 是否仅评估模式
                    help='仅评估模式：构建dataloader但不进入训练循环（默认关闭）')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu") # 有标签数据的批处理大小(batch size)
parser.add_argument("--unlabel_bs", type=int, default=4) # 无标签数据的批处理大小(batch size)
parser.add_argument("--test_bs", type=int, default=1) # 测试时的批处理大小
parser.add_argument('--domain_num', type=int, default=6) # 数据集中域(domain)的总数量
parser.add_argument('--lb_domain', type=int, default=1) # 指定哪个域作为有标签数据的来源
parser.add_argument('--lb_num', type=int, default=40) # 使用的有标签样本数量
parser.add_argument('--lb_ratio', type=float, default=0) # 有标签样本比例（相对于总样本数）；若>0则按比例计算 lb_num，优先级高于 lb_num
# 一致性系数与ramp-up（论文式(2) λ(t) 类似暖启动思想；此处使用 ramps.sigmoid_rampup 实现随迭代增长的权重）
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay") # EMA（指数移动平均）更新教师模型的衰减率，用于Mean Teacher架构
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type") # 一致性损失的类型（例如mse）
parser.add_argument("--consistency", type=float, default=1.0, help="consistency") # 一致性损失项的权重
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup") # 一致性权重从0增长到最大值的ramp-up阶段长度，用于平滑引入无监督损失

# U-Net 超参（若后续扩展 WRN 等）
parser.add_argument('--depth', type=int, default=28) # WRN 深度（未使用）
parser.add_argument('--widen_factor', type=int, default=2) # WRN 宽度因子（未使用）
parser.add_argument('--leaky_slope', type=float, default=0.1) # WRN LeakyReLU 负斜率（未使用）
parser.add_argument('--bn_momentum', type=float, default=0.1) # 批归一化动量（未使用）
parser.add_argument('--dropout', type=float, default=0.0) # WRN dropout 比例（未使用）
parser.add_argument('--save_img',action='store_true') # 是否保存预测图像
parser.add_argument('--save_model',action='store_true') # 是否保存模型
# LoRA 相关：论文在 MedSAM 上采用 LoRA 适配器以减少计算开销（冻结图像编码器，训练低秩适配/解码器）【§3.3】:contentReference[oaicite:7]{index=7}
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation') # LoRA（Low-Rank Adaptation）的秩(rank)，用于高效微调大模型
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr') # 是否启用学习率预热（从较低学习率逐步升至基础学习率）
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated') # 学习率预热的迭代次数
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model') # 是否使用 AdamW 优化器来微调 SAM 模型
parser.add_argument('--module', type=str, default='sam_lora_image_encoder') # 动态导入 LoRA 模块的名称（需与文件名对应）
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input') # 基础模型（如MedSAM）的输入图像尺寸
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model') # 选择ViT模型的版本，例如 "vit_b"
# parser.add_argument('--ckpt', type=str, default='../checkpoints/sam_vit_b_01ec64.pth',
#                     help='Pretrained checkpoint')
parser.add_argument('--ckpt', type=str, default='../checkpoints/medsam_vit_b.pth',
                    help='Pretrained checkpoint') # 预训练检查点路径（MedSAM 权重）

parser.add_argument('--output_root', type=str, default='../model',
                    help='保存模型和日志的根目录') # 新增！保存模型和日志的根目录
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    """
    一致性损失权重随训练迭代进行“sigmoid ramp-up”增长（与论文式(2) λ(t) 的“时间相关加权/预热”理念一致）:contentReference[oaicite:8]{index=8}
    目的：前期避免不可靠伪标签/一致性约束干扰监督学习，后期逐步增强无监督一致性/正则约束。
    """
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_Unet_ema_variables(model, ema_model, alpha, global_step):
    """
    Mean Teacher：用学生参数对教师(U-Net EMA)做指数滑动平均（论文 §3.1 训练范式，“教师为学生EMA”）:contentReference[oaicite:9]{index=9}
    alpha：EMA 衰减；前期用真实平均(1-1/(step+1))，后期固定 alpha。
    """
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def update_SAM_ema_variables(model, ema_model, alpha, global_step):
    """
    对 MedSAM 的 EMA：由于采用 LoRA + 解码/提示编码器可训练参数，
    这里分别对 LoRA A/B 低秩层以及 mask/prompt 解码/编码器做 EMA 更新（论文 §3.3）:contentReference[oaicite:10]{index=10}
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_A_linear, A_linear in zip(ema_model.w_As, model.w_As):
        ema_A_linear.weight.data.mul_(alpha).add_(A_linear.weight.data, alpha=1 - alpha)
    for ema_B_linear, B_linear in zip(ema_model.w_Bs, model.w_Bs):
        ema_B_linear.weight.data.mul_(alpha).add_(B_linear.weight.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.sam.mask_decoder.parameters(), model.sam.mask_decoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.sam.prompt_encoder.parameters(), model.sam.prompt_encoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """
    安全的“无限数据流”封装：避免 itertools.cycle 对 DataLoader(shuffle=True) 仅首轮随机的问题。
    每次循环重新遍历可确保每个 epoch 的随机性。详见注释链接。
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    """
    便捷构建优化器（按 BN 是否跳过 weight decay 拆参数组）。
    """
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer

# @torch.no_grad()
# def test(args, model, test_dataloader, epoch, writer, model_name):
#     """
#     评估：逐域(test_dataloader 列表包含每个域的 DataLoader)，输出：
#     - loss（其实此处 val_loss=0，仅示例保留）
#     - Dice / DC, JC, HD95, ASD（每个 part/器官/前景类别分别统计；论文 §4.1 指标一致）:contentReference[oaicite:11]{index=11}
#     注意：根据数据集差异，label 语义映射不同（fundus/prostate/MNMS/BUSI）。
#     """
#     model.eval()
#     val_loss = 0.0
#     val_dice = [0.0] * n_part
#     val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
#     domain_num = len(test_dataloader)
#     ce_loss = CrossEntropyLoss(reduction='none')
#     softmax, sigmoid, multi = True, False, False
#     dice_loss = losses.DiceLossWithMask(2)
#     for i in range(domain_num):
#         cur_dataloader = test_dataloader[i]
#         domain_val_loss = 0.0
#         domain_val_dice = [0.0] * n_part
#         domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
#         domain_code = i+1
#         for batch_num,sample in enumerate(cur_dataloader):
#             assert(domain_code == sample['dc'][0].item())
#             mask = sample['label']
#             # --------- 不同数据集的label预处理（把原始像素值映射到{0,1,...,C}）---------
#             if args.dataset == 'fundus':
#                 # fundus 原掩膜像素：0/<=128/>128，转为两类 one-hot 的索引（1/2），背景最终仍处理到二维 one-hot
#                 lb_mask = (mask<=128) * 2
#                 lb_mask[mask==0] = 1
#                 mask = lb_mask
#             elif args.dataset == 'prostate':
#                 mask = mask.eq(0).long()
#             elif args.dataset == 'MNMS':
#                 mask = mask.long()
#             elif args.dataset == 'BUSI':
#                 mask = mask.eq(255).long()
#             # --------- 前向：区分 SAM 分支与 U-Net 分支 -----------
#             if model_name == 'SAM':
#                 data = sample['image'].cuda()                # SAM 输入：上采样到 512（论文 §3.5）:contentReference[oaicite:12]{index=12}
#                 output = model(data, multimask_output, args.img_size)['masks']
#             elif model_name == 'unet':
#                 data = sample['unet_size_img'].cuda()        # U-Net 输入：数据集原设定 patch_size（W×H）
#                 output = model(data)
#             pred_label = torch.max(torch.softmax(output,dim=1), dim=1)[1]
#             # 将预测 resize 到统一 patch_size 再评估（各分支输出分辨率不同，论文 §3.5）:contentReference[oaicite:13]{index=13}
#             pred_label = torch.from_numpy(zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0))
            
#             # --------- 为几何指标构造 one-hot 或多通道布尔掩膜 ----------
#             if args.dataset == 'fundus':
#                 pred_label = to_2d(pred_label)
#                 mask = to_2d(mask)
#                 pred_onehot = pred_label.clone()
#                 mask_onehot = mask.clone()
#             elif args.dataset == 'prostate' or args.dataset == 'BUSI':
#                 pred_onehot = pred_label.clone().unsqueeze(1)
#                 mask_onehot = mask.clone().unsqueeze(1)
#             elif args.dataset == 'MNMS':
#                 pred_onehot = to_3d(pred_label)
#                 mask_onehot = to_3d(mask)
#             dice = dice_calcu[args.dataset](np.asarray(pred_label.cpu()),mask.cpu())
            
#             # 逐 batch 统计 DC/JC/HD95/ASD（使用 medpy.metric.binary）：
#             dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
#             for j in range(len(data)):
#                 for i, p in enumerate(part):
#                     dc[i] += binary.dc(np.asarray(pred_onehot[j,i], dtype=bool),
#                                             np.asarray(mask_onehot[j,i], dtype=bool))
#                     jc[i] += binary.jc(np.asarray(pred_onehot[j,i], dtype=bool),
#                                             np.asarray(mask_onehot[j,i], dtype=bool))
#                     if pred_onehot[j,i].float().sum() < 1e-4:
#                         hd[i] += 100
#                         asd[i] += 100
#                     else:
#                         hd[i] += binary.hd95(np.asarray(pred_onehot[j,i], dtype=bool),
#                                             np.asarray(mask_onehot[j,i], dtype=bool))
#                         asd[i] += binary.asd(np.asarray(pred_onehot[j,i], dtype=bool),
#                                             np.asarray(mask_onehot[j,i], dtype=bool))
#             for i, p in enumerate(part):
#                 dc[i] /= len(data)
#                 jc[i] /= len(data)
#                 hd[i] /= len(data)
#                 asd[i] /= len(data)
#             for i in range(len(domain_val_dice)):
#                 domain_val_dice[i] += dice[i]
#                 domain_val_dc[i] += dc[i]
#                 domain_val_jc[i] += jc[i]
#                 domain_val_hd[i] += hd[i]
#                 domain_val_asd[i] += asd[i]
        
#         domain_val_loss /= len(cur_dataloader)
#         val_loss += domain_val_loss
#         writer.add_scalar('{}_val/domain{}/loss'.format(model_name, domain_code), domain_val_loss, epoch)
#         for i in range(len(domain_val_dice)):
#             domain_val_dice[i] /= len(cur_dataloader)
#             val_dice[i] += domain_val_dice[i]
#             domain_val_dc[i] /= len(cur_dataloader)
#             val_dc[i] += domain_val_dc[i]
#             domain_val_jc[i] /= len(cur_dataloader)
#             val_jc[i] += domain_val_jc[i]
#             domain_val_hd[i] /= len(cur_dataloader)
#             val_hd[i] += domain_val_hd[i]
#             domain_val_asd[i] /= len(cur_dataloader)
#             val_asd[i] += domain_val_asd[i]
#         for n, p in enumerate(part):
#             writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, domain_code, p), domain_val_dice[n], epoch)
#         text = 'domain%d epoch %d : loss : %f' % (domain_code, epoch, domain_val_loss)
@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, model_name):
    model.eval()
    val_loss = 0.0
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)

    # --- loss 定义（与训练保持一致） ---
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes+1)  # NEW：用全局 num_classes

    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i+1

        for batch_num,sample in enumerate(cur_dataloader):
            assert(domain_code == sample['dc'][0].item())

            # --------- 读取不同分辨率的标签，用于各分支计算损失（NEW）---------
            # 原始 label 仍用于几何指标；各数据集做同样的映射（与训练一致）
            raw_mask = sample['label']

            if args.dataset == 'fundus':
                mask_metrics = (raw_mask<=128) * 2
                mask_metrics[raw_mask==0] = 1

                low_res_mask = (sample['low_res_label']<=128) * 2
                low_res_mask[sample['low_res_label']==0] = 1

                unet_size_mask = (sample['unet_size_label']<=128) * 2
                unet_size_mask[sample['unet_size_label']==0] = 1

            elif args.dataset == 'prostate':
                mask_metrics = raw_mask.eq(0).long()
                low_res_mask = sample['low_res_label'].eq(0).long()
                unet_size_mask = sample['unet_size_label'].eq(0).long()

            elif args.dataset == 'MNMS':
                mask_metrics = raw_mask.long()
                low_res_mask = sample['low_res_label'].long()
                unet_size_mask = sample['unet_size_label'].long()

            elif args.dataset == 'BUSI':
                mask_metrics = raw_mask.eq(255).long()
                low_res_mask = sample['low_res_label'].eq(255).long()
                unet_size_mask = sample['unet_size_label'].eq(255).long()

            # --------- 前向（分支各自使用匹配分辨率）-----------
            if model_name == 'SAM':
                data = sample['image'].cuda()
                # NEW：拿低分辨率 logits 来算 CE/Dice，而不是 ['masks']
                out = model(data, multimask_output, args.img_size)
                logits = out['low_res_logits']            # [B, C, low_res, low_res]
                label4loss = low_res_mask.cuda()          # [B, H_low, W_low]
                # 预测（仅用于指标）
                pred_prob = torch.softmax(logits, dim=1)
                pred_label = torch.max(pred_prob, dim=1)[1]
                # 为几何指标 resize 到统一 patch_size（保持原逻辑）
                pred_label = torch.from_numpy(
                    zoom(pred_label.cpu(), (1, patch_size / logits.shape[-2], patch_size / logits.shape[-1]), order=0)
                )
            elif model_name == 'unet':
                data = sample['unet_size_img'].cuda()
                logits = model(data)                       # [B, C, H_unet, W_unet]
                label4loss = unet_size_mask.cuda()
                pred_prob = torch.softmax(logits, dim=1)
                pred_label = torch.max(pred_prob, dim=1)[1]
                pred_label = torch.from_numpy(
                    zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0)
                )
            else:
                raise ValueError

            # --------- 计算并累加 “验证损失”（CE + Dice）---------
            # （注意：验证阶段不做阈值掩蔽，直接全图范围）
            batch_ce = ce_loss(logits, label4loss).mean()
            batch_dice = dice_loss(logits, label4loss.unsqueeze(1), softmax=True, sigmoid=False, multi=False)
            batch_loss = batch_ce + batch_dice
            domain_val_loss += batch_loss.item()          # NEW：累计域内损失

            # --------- 指标（保持你原先的计算流程）---------
            if args.dataset == 'fundus':
                pred_label = to_2d(pred_label)
                mask4metrics = to_2d(mask_metrics)
                pred_onehot = pred_label.clone()
                mask_onehot = mask4metrics.clone()
            elif args.dataset in ['prostate', 'BUSI']:
                mask4metrics = mask_metrics
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask4metrics.clone().unsqueeze(1)
            elif args.dataset == 'MNMS':
                mask4metrics = mask_metrics
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask4metrics)

            dice = dice_calcu[args.dataset](np.asarray(pred_label.cpu()), mask4metrics.cpu())

            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for ii, p in enumerate(part):
                    dc[ii] += binary.dc(np.asarray(pred_onehot[j,ii], dtype=bool),
                                        np.asarray(mask_onehot[j,ii], dtype=bool))
                    jc[ii] += binary.jc(np.asarray(pred_onehot[j,ii], dtype=bool),
                                        np.asarray(mask_onehot[j,ii], dtype=bool))
                    if pred_onehot[j,ii].float().sum() < 1e-4:
                        hd[ii] += 100
                        asd[ii] += 100
                    else:
                        hd[ii] += binary.hd95(np.asarray(pred_onehot[j,ii], dtype=bool),
                                              np.asarray(mask_onehot[j,ii], dtype=bool))
                        asd[ii] += binary.asd(np.asarray(pred_onehot[j,ii], dtype=bool),
                                              np.asarray(mask_onehot[j,ii], dtype=bool))
            for ii in range(len(domain_val_dice)):
                dc[ii] /= len(data); jc[ii] /= len(data); hd[ii] /= len(data); asd[ii] /= len(data)

            for ii in range(len(domain_val_dice)):
                domain_val_dice[ii] += dice[ii]
                domain_val_dc[ii]   += dc[ii]
                domain_val_jc[ii]   += jc[ii]
                domain_val_hd[ii]   += hd[ii]
                domain_val_asd[ii]  += asd[ii]

        # --- 归一化并写入日志 ---
        domain_val_loss /= max(1, len(cur_dataloader))   # NEW：求平均
        val_loss += domain_val_loss
        writer.add_scalar(f'{model_name}_val/domain{domain_code}/loss', domain_val_loss, epoch)

        for ii in range(len(domain_val_dice)):
            domain_val_dice[ii] /= len(cur_dataloader)
            val_dice[ii] += domain_val_dice[ii]
            domain_val_dc[ii] /= len(cur_dataloader); val_dc[ii] += domain_val_dc[ii]
            domain_val_jc[ii] /= len(cur_dataloader); val_jc[ii] += domain_val_jc[ii]
            domain_val_hd[ii] /= len(cur_dataloader); val_hd[ii] += domain_val_hd[ii]
            domain_val_asd[ii] /= len(cur_dataloader); val_asd[ii] += domain_val_asd[ii]

        # 原有日志字符串保持，只是现在 loss 不再是 0 了
        text = 'domain%d epoch %d : loss : %f' % (domain_code, epoch, domain_val_loss)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
    logging.info(text)
    return val_dice
    
def to_2d(input_tensor):
    """
    fundus 二类任务的 one-hot 展开：将整数标签 -> 两通道布尔（背景/前景）
    """
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def to_3d(input_tensor):
    """
    MNMS 三类任务的 one-hot 展开：将整数标签 -> 三通道布尔（LV/MYO/RV）
    """
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    """
    生成 Copy-Paste/混合区域 M（矩形随机窗口），概率 p 触发。
    论文式(1)中的 M 即此（将有标签弱增强图像 X_w 部分粘贴到 U_s 上，构造中间样本 U_c 与伪标签 Q_c）。:contentReference[oaicite:14]{index=14}
    """
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

class statistics(object):
    """简单统计器：用于滚动记录(均值显示在日志/进度条)"""
    def __init__(self):
        self.record = []
        self.num = 0
        self.avg = 0

    def update(self, val):
        self.record.append(val)
        self.num += 1
        self.avg = sum(self.record) / self.num

def train(args, snapshot_path):
    """
    训练主函数：构建 MedSAM(带 LoRA) 与 U-Net 两个学生模型 + 对应 EMA 教师模型；
    数据：带标签(lb)与无标签(ulb)，并按论文 MiDSS 多域设置构建测试集（§4.1）。
    核心逻辑：
      - 教师(EMA)对无标签弱增强图像 U_w 产生伪标签；（§3.1）
      - 通过 Copy-Paste 生成中间样本 U_c 与对应伪标签；（式(1)）
      - SMC：计算 U-Net 自信(teacher vs student)/互信(teacher_UT vs teacher_MS)得到 α；（式(4)(5)(6)）
      - 伪标签集成：\hat{P}_w^{EN} = α * UT + (1-α) * MS；（式(3)）
      - 阈值过滤高置信区域形成 mask，用于无监督 CE+Dice；（式(7)）
      - CDCR：一致区域最小化熵、发散区域最小化 MSE（consistency + divergence）；（式(9)(10)）
      - 总损失：监督(sup)+一致性/无监督等按 ramp-up 加权（理念对应式(2)）。:contentReference[oaicite:15]{index=15}
    """
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    def create_model(model_name=None, ema=False):
        # 根据名称构建模型与其 EMA（教师模型只做参数追踪，不反传）
        if model_name == 'SAM':
            logging.info("load from {}".format(args.ckpt))
            # 构建 MedSAM (基于 SAM 注册表)，设置 num_classes 与图像统计
            sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
            # 动态导入 LoRA 包并包装 SAM（论文 §3.3：对图像编码器使用 LoRA，训练 LoRA+解码器）:contentReference[oaicite:16]{index=16}
            pkg = import_module(args.module)
            model = pkg.LoRA_Sam(sam, args.rank)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model.cuda(), img_embedding_size
        elif model_name == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes+1)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model.cuda()
        else:
            raise Exception('Please provide model name.')

    # 学生/教师模型（SAM & U-Net）
    SAM_model, img_embedding_size = create_model(model_name='SAM')
    ema_SAM_model, _ = create_model(model_name='SAM', ema=True)
    unet_model = create_model(model_name='unet')
    ema_unet_model = create_model(model_name='unet', ema=True)
    
    # SAM 低分辨率 logits 边长（论文 §3.5：SAM 输出128×128；此处由 img_embedding_size * 4 给定下游上采/下采尺寸关系）:contentReference[oaicite:17]{index=17}
    low_res = img_embedding_size * 4

    max_iterations = args.max_iterations
    # --------- 弱增强（对 lb/ulb 均使用；论文 §3.1）---------
    weak = transforms.Compose([tr.RandomScaleCrop(args.img_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform()
            ])
    # --------- 强增强（对 ulb 的 U_s）---------
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * args.img_size), num_channels=num_channels),
    ])
    # --------- 归一化+ToTensor+尺寸对齐（SAM 与 U-Net 双分支需求）---------
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(low_res=low_res, unet_size=patch_size)
    ])

    # ----------------- 数据组织（MiDSS 多域）-----------------
    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    # 各数据集每域样本数（用于划分 lb/ulb 索引）
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133]
    elif args.dataset == 'MNMS':
        domain_len = [1030, 1342, 525, 550]
    elif args.dataset == 'BUSI':
        domain_len = [350, 168]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    if args.lb_ratio > 0:
        lb_num = int(sum(domain_len) * args.lb_ratio)
    else:
        lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    # 标注集（来自单一域 lb_domain）：论文 MiDSS 设置（§4.1）:contentReference[oaicite:18]{index=18}
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor, img_size=args.img_size)
    # 未标注集（混合多域）：论文 MiDSS 设置（§4.1）
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor, img_size=args.img_size)
    # 测试集：每个域一个 DataLoader（用于报告跨域平均）
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor, img_size=args.img_size)
        test_dataset.append(cur_dataset)
    if not args.eval:
        # 使用 cycle() 保证 shuffle 后每 epoch 都能均匀抽取
        lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
        ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    iter_num = 0
    start_epoch = 0

    # ----------------- 损失与优化器设置 -----------------
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    # DiceLossWithMask：支持在 mask 区域计算（用于 τ 阈值过滤的高置信区域），对应论文式(7)权重 Wc^EN。:contentReference[oaicite:19]{index=19}
    dice_loss = losses.DiceLossWithMask(num_classes+1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    # MedSAM 优化器：可选 AdamW（论文 §4.2 默认 AdamW, lr=1e-4, wd=0.1），或 SGD（注：源码给出两种备选）:contentReference[oaicite:20]{index=20}
    if args.AdamW:
        sam_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, SAM_model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        sam_optimizer = optim.SGD(filter(lambda p: p.requires_grad, SAM_model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # 原来写成了 model.parameters()，会 NameError —— 改为 SAM_model.parameters()
    # U-Net 优化器：SGD（论文 §4.2 超参对齐）:contentReference[oaicite:21]{index=21}
    unet_optimizer = optim.SGD(unet_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    # 记录最好指标（均为 per-part 与平均）
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)

    threshold = args.threshold

    # AMP 混合精度
    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    for epoch_num in range(start_epoch, max_epoch):
        SAM_model.train()
        ema_SAM_model.train()
        unet_model.train()
        ema_unet_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        # 统计 SMC 与各自/融合伪标签质量（论文图6类似曲线展示）:contentReference[oaicite:22]{index=22}
        self_conf_sta, mutual_conf_sta, ratio_sta, SAM_ulb_dice_sta, unet_ulb_dice_sta, ensemble_ulb_dice_sta = statistics(), statistics(), statistics(), [statistics() for _ in range(n_part)], [statistics() for _ in range(n_part)], [statistics() for _ in range(n_part)]
        for i_batch in range(1, args.num_eval_iter+1):
            # ----- 取一个 lb 批与一个 ulb 批 -----
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']                            # X_w, Y_w（弱增强）
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']  # U_w, U_s（弱/强增强）
            lb_low_res_y, ulb_low_res_y = lb_sample['low_res_label'], ulb_sample['low_res_label']          # 对齐 SAM 低分辨率分支的标签
            lb_unet_size_x_w, lb_unet_size_y = lb_sample['unet_size_img'], lb_sample['unet_size_label']    # 对齐 U-Net 分支的输入/标签尺寸
            ulb_unet_size_x_w, ulb_unet_size_x_s, ulb_unet_size_y = ulb_sample['unet_size_img'], ulb_sample['unet_size_strong_aug'], ulb_sample['unet_size_label']
            
            # ---------- 标注与未标注的标签值映射（同 test()） ----------
            if args.dataset == 'fundus':
                lb_mask = (lb_y<=128) * 2
                lb_mask[lb_y==0] = 1
                ulb_mask = (ulb_y<=128) * 2
                ulb_mask[ulb_y==0] = 1
                lb_low_res_mask = (lb_low_res_y<=128) * 2
                lb_low_res_mask[lb_low_res_y==0] = 1
                ulb_low_res_mask = (ulb_low_res_y<=128) * 2
                ulb_low_res_mask[ulb_low_res_y==0] = 1
                lb_unet_size_mask = (lb_unet_size_y<=128) * 2
                lb_unet_size_mask[lb_unet_size_y==0] = 1
                ulb_unet_size_mask = (ulb_unet_size_y<=128) * 2
                ulb_unet_size_mask[ulb_unet_size_y==0] = 1
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()
                lb_low_res_mask = lb_low_res_y.eq(0).long()
                ulb_low_res_mask = ulb_low_res_y.eq(0).long()
                lb_unet_size_mask = lb_unet_size_y.eq(0).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(0).long()
            elif args.dataset == 'MNMS':
                lb_mask = lb_y.long()
                ulb_mask = ulb_y.long()
                lb_low_res_mask = lb_low_res_y.long()
                ulb_low_res_mask = ulb_low_res_y.long()
                lb_unet_size_mask = lb_unet_size_y.long()
                ulb_unet_size_mask = ulb_unet_size_y.long()
            elif args.dataset == 'BUSI':
                lb_mask = lb_y.eq(255).long()
                ulb_mask = ulb_y.eq(255).long()
                lb_low_res_mask = lb_low_res_y.eq(255).long()
                ulb_low_res_mask = ulb_low_res_y.eq(255).long()
                lb_unet_size_mask = lb_unet_size_y.eq(255).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(255).long()
            # 送 GPU
            lb_x_w, lb_mask, ulb_x_w, ulb_x_s, ulb_mask = lb_x_w.cuda(), lb_mask.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_mask.cuda()
            lb_unet_size_x_w, ulb_unet_size_x_w, ulb_unet_size_x_s = lb_unet_size_x_w.cuda(), ulb_unet_size_x_w.cuda(), ulb_unet_size_x_s.cuda()
            lb_low_res_mask, ulb_low_res_mask = lb_low_res_mask.cuda(), ulb_low_res_mask.cuda()
            lb_unet_size_mask, ulb_unet_size_mask = lb_unet_size_mask.cuda(), ulb_unet_size_mask.cuda()

            with amp_cm():
                # -------------------- 教师模型对 U_w 推理，得到各自分支的伪标签（论文 §3.3） --------------------
                with torch.no_grad():
                    # MedSAM 教师 on U_w -> 低分辨率 logits/prob
                    sam_output_ulb_x_w = ema_SAM_model(ulb_x_w, multimask_output, args.img_size)
                    sam_logits_ulb_x_w = sam_output_ulb_x_w['low_res_logits']
                    sam_prob_ulb_x_w = torch.softmax(sam_logits_ulb_x_w, dim=1)
                    sam_prob, sam_pseudo_label = torch.max(sam_prob_ulb_x_w, dim=1)
                    # 将 SAM 概率上采到 U-Net 尺寸，便于后续融合/一致性度量
                    unet_size_sam_prob_ulb_x_w = F.interpolate(sam_prob_ulb_x_w, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    # U-Net 教师 on U_w (unet-size)
                    unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
                    unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
                    unet_prob, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)
                    
                # -------------------- 计算 SMC 自/互信度与 α（论文 式(4)(5)(6)） --------------------
                # U-Net 学生自输出（与自身教师输出比较 -> 自信度 Φ^self）
                unet_stu_output_ulb_x_w = unet_model(ulb_unet_size_x_w)
                unet_stu_prob_ulb_x_w = torch.softmax(unet_stu_output_ulb_x_w, dim=1)
                unet_stu_prob, unet_stu_pseudo_label = torch.max(unet_stu_prob_ulb_x_w, dim=1)
                # 将 U-Net 教师伪标签降采到 SAM 低分辨率，用于与 SAM 教师伪标签比较 -> 互信度 Φ^mut
                low_res_unet_pseudo_label = F.interpolate(unet_pseudo_label.unsqueeze(0).float(), size=(low_res, low_res), mode='nearest').long().squeeze(0)
                if args.dataset == 'fundus':
                    # 二类任务的 dice 计算做了 2-layer 展开
                    self_conf = dice_calcu[args.dataset](np.asarray(to_2d(unet_stu_pseudo_label).cpu()), to_2d(unet_pseudo_label).cpu(), ret_arr=True)
                    mutual_conf = dice_calcu[args.dataset](np.asarray(to_2d(low_res_unet_pseudo_label).cpu()), to_2d(sam_pseudo_label).cpu(), ret_arr=True)
                else:
                    self_conf = dice_calcu[args.dataset](np.asarray(unet_stu_pseudo_label.clone().cpu()), unet_pseudo_label.clone().cpu(), ret_arr=True)
                    mutual_conf = dice_calcu[args.dataset](np.asarray(low_res_unet_pseudo_label.clone().cpu()), sam_pseudo_label.clone().cpu(), ret_arr=True)
                self_conf, mutual_conf = np.mean(self_conf, axis=0), np.mean(mutual_conf, axis=0)
                
                # 日志统计：平均自信、互信与二者乘积（论文图6展示 α 与伪标签质量随时间变化关系）:contentReference[oaicite:23]{index=23}
                self_conf_sta.update(np.mean(self_conf))
                mutual_conf_sta.update(np.mean(mutual_conf))
                ratio_sta.update(np.mean(self_conf * mutual_conf))
                
                # α = Φ^self × Φ^mut（式(6)）；将其作为权重融合两分支概率（式(3)）
                ratio =  torch.tensor(self_conf * mutual_conf).view(len(ulb_x_w),1,1,1).cuda()
                unet_size_prob_ulb_x_w = (1-ratio)*unet_size_sam_prob_ulb_x_w + ratio*unet_prob_ulb_x_w
                unet_size_prob, unet_size_pseudo_label = torch.max(unet_size_prob_ulb_x_w, dim=1)
                # 高置信遮罩（> τ）：用于无监督损失掩蔽（式(7)中的 W_c^EN）
                unet_size_mask = (unet_size_prob > threshold).unsqueeze(1).float()
                # 同步生成低分辨率版（SAM 支路用）
                low_res_prob_ulb_x_w = F.interpolate(unet_size_prob_ulb_x_w, size=(low_res, low_res), mode='bilinear', align_corners=False)
                low_res_prob, low_res_pseudo_label = torch.max(low_res_prob_ulb_x_w, dim=1)
                low_res_mask = (low_res_prob > threshold).unsqueeze(1).float()
                
                # -------------------- 生成中间样本 U_c（式(1)）并混合标签 --------------------
                # 在 U-Net 尺寸空间生成剪贴框，并映射到 SAM 低分辨率与原图尺寸
                unet_size_label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=1.0) for i in range(len(ulb_x_s))], dim=0)
                unet_size_img_box = unet_size_label_box.unsqueeze(1)
                img_box = F.interpolate(unet_size_img_box, size=(args.img_size, args.img_size), mode='nearest')
                low_res_img_box = F.interpolate(unet_size_img_box, size=(low_res, low_res), mode='nearest')
                low_res_label_box = low_res_img_box.squeeze(1)
                # 构造 U_c 图像（对 ulb 强增强图像与 lb 弱增强图像进行区域级粘贴）
                ulb_unet_size_x_s_ul = ulb_unet_size_x_s * (1-unet_size_img_box) + lb_unet_size_x_w * unet_size_img_box
                ulb_x_s_ul = ulb_x_s * (1-img_box) + lb_x_w * img_box
                # 将被粘贴区域对应的 mask 置1，确保这些区域的伪标签/监督不被过滤掉（与式(1)对齐）
                unet_size_mask[unet_size_img_box.expand(unet_size_mask.shape) == 1] = 1
                low_res_mask[low_res_img_box.expand(low_res_mask.shape) == 1] = 1
                # 构造中间样本的伪标签（把 lb 的真值粘贴到选定区域，其余为融合伪标签）
                low_res_pseudo_label_ul = (low_res_pseudo_label * (1-low_res_label_box) + lb_low_res_mask * low_res_label_box).long()
                unet_size_pseudo_label_ul = (unet_size_pseudo_label * (1-unet_size_label_box) + lb_unet_size_mask * unet_size_label_box).long()
                
                # -------------------- 学生模型在 U_c 上前向，用于无监督一致性学习 --------------------
                sam_output_lb_x_w = SAM_model(lb_x_w, multimask_output, args.img_size)
                sam_logits_lb_x_w = sam_output_lb_x_w['low_res_logits']
                sam_output_ulb_x_s_ul = SAM_model(ulb_x_s_ul, multimask_output, args.img_size)
                sam_logits_ulb_x_s_ul = sam_output_ulb_x_s_ul['low_res_logits']
                unet_logits_lb_x_w = unet_model(lb_unet_size_x_w)
                unet_logits_ulb_x_s_ul = unet_model(ulb_unet_size_x_s_ul)
                
                # -------------------- CDCR：共识/发散一致性正则（论文 §3.4，式(9)(10)） --------------------
                # 首先确定一致/不一致区域：通过学生两分支伪标签是否一致判断
                sam_prob_ulb_x_s_ul = torch.softmax(sam_logits_ulb_x_s_ul, dim=1)
                unet_size_sam_prob_ulb_x_s_ul = F.interpolate(sam_prob_ulb_x_s_ul, size=(patch_size, patch_size), mode='nearest')
                unet_prob_ulb_x_s_ul = torch.softmax(unet_logits_ulb_x_s_ul, dim=1)
                _, sam_PL_stu = torch.max(unet_size_sam_prob_ulb_x_s_ul, dim=1)
                _, unet_PL_stu = torch.max(unet_prob_ulb_x_s_ul, dim=1)
                cons_mask = (sam_PL_stu == unet_PL_stu).unsqueeze(1).float()          # M_c（式(8)）
                low_res_cons_mask = F.interpolate(cons_mask, size=(low_res, low_res), mode='nearest')
                discons_mask = 1-cons_mask                                           # M_d = 1 - M_c
                low_res_discons_mask = 1-low_res_cons_mask
                # 共识区：最小化熵（鼓励高置信、低熵） -> -p*log p（式(9)）
                epsilon = 1e-10
                sam_prob_ulb_x_s_ul = torch.clamp(sam_prob_ulb_x_s_ul, epsilon, 1)
                unet_prob_ulb_x_s_ul = torch.clamp(unet_prob_ulb_x_s_ul, epsilon, 1)
                cons_loss = (-(sam_prob_ulb_x_s_ul*torch.log(sam_prob_ulb_x_s_ul)*low_res_cons_mask).mean()-(unet_prob_ulb_x_s_ul*torch.log(unet_prob_ulb_x_s_ul)*cons_mask).mean())/2
                # 发散区：最小化 MSE（拉近两分支分布） -> (p_ms - p_ut)^2（式(10)）
                discons_loss = ((unet_size_sam_prob_ulb_x_s_ul-unet_prob_ulb_x_s_ul)**2*discons_mask).mean()

                # 统计：各分支与融合伪标签在 ulb 上的 Dice（论文 图3/图6 中展示对比）:contentReference[oaicite:24]{index=24}
                if args.dataset == 'fundus':
                    sam_pseudo_label_2layer = to_2d(sam_pseudo_label)
                    unet_pseudo_label_2layer = to_2d(unet_pseudo_label)
                    pseudo_label_2layer = to_2d(unet_size_pseudo_label)
                    ulb_low_res_mask_2layer = to_2d(ulb_low_res_mask)
                    ulb_unet_size_mask_2layer = to_2d(ulb_unet_size_mask)
                    sam_ulb_dice = dice_calcu[args.dataset](np.asarray(sam_pseudo_label_2layer.cpu()), ulb_low_res_mask_2layer.cpu())
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                    ulb_dice = dice_calcu[args.dataset](np.asarray(pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                else:
                    sam_ulb_dice = dice_calcu[args.dataset](np.asarray(sam_pseudo_label.cpu()), ulb_low_res_mask.cpu())
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                    ulb_dice = dice_calcu[args.dataset](np.asarray(unet_size_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                for n, p in enumerate(part):
                    unet_ulb_dice_sta[n].update(unet_ulb_dice[n])
                    SAM_ulb_dice_sta[n].update(sam_ulb_dice[n])
                    ensemble_ulb_dice_sta[n].update(ulb_dice[n])

                # -------------------- 监督损失（lb）：CE + Dice --------------------
                # 两分支各自对其标注视图计算监督损失 L_x（论文式(2)第一项）:contentReference[oaicite:25]{index=25}
                sam_sup_loss = ce_loss(sam_logits_lb_x_w, lb_low_res_mask).mean() + \
                            dice_loss(sam_logits_lb_x_w, lb_low_res_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                unet_sup_loss = ce_loss(unet_logits_lb_x_w, lb_unet_size_mask).mean() + \
                            dice_loss(unet_logits_lb_x_w, lb_unet_size_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                # ramp-up 一致性权重（理念对应论文式(2) λ(t)）
                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                # -------------------- 无监督损失（U_c）：阈值过滤的 CE + Dice（式(7)） --------------------
                sam_unsup_loss = (ce_loss(sam_logits_ulb_x_s_ul, low_res_pseudo_label_ul) * low_res_mask.squeeze(1)).mean() + \
                                dice_loss(sam_logits_ulb_x_s_ul, low_res_pseudo_label_ul.unsqueeze(1), mask=low_res_mask, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unet_unsup_loss = (ce_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul) * unet_size_mask.squeeze(1)).mean() + \
                                dice_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul.unsqueeze(1), mask=unet_size_mask, softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                # -------------------- 总损失：监督 + λ*(无监督 + CDCR)（对应式(2)） --------------------
                loss = sam_sup_loss + unet_sup_loss + consistency_weight * (sam_unsup_loss + unet_unsup_loss+cons_loss+discons_loss)

            # 反传/更新
            sam_optimizer.zero_grad()
            unet_optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(sam_optimizer)
                scaler.step(unet_optimizer)
                scaler.update()
            else:
                loss.backward()
                sam_optimizer.step()
                unet_optimizer.step()

            # EMA 更新教师参数（§3.1）
            update_SAM_ema_variables(SAM_model, ema_SAM_model, args.ema_decay, iter_num)
            update_Unet_ema_variables(unet_model, ema_unet_model, args.ema_decay, iter_num)

            # 迭代计数与日志
            iter_num = iter_num + 1
            for n, p in enumerate(part):
                text = 'train/unet_ulb_{}_dice'.format(p)
                writer.add_scalar(text, unet_ulb_dice[n], iter_num)
            for n, p in enumerate(part):
                text = 'train/sam_ulb_{}_dice'.format(p)
                writer.add_scalar(text, sam_ulb_dice[n], iter_num)
            for n, p in enumerate(part):
                text = 'train/ensemble_ulb_{}_dice'.format(p)
                writer.add_scalar(text, ulb_dice[n], iter_num)
            writer.add_scalar('train/self_conf', np.mean(self_conf), iter_num)
            writer.add_scalar('train/mutual_conf', np.mean(mutual_conf), iter_num)
            writer.add_scalar('train/ratio', np.mean(self_conf*mutual_conf), iter_num)  # α 的期望（图6）
            writer.add_scalar('train/mask', unet_size_mask.mean(), iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sam_sup_loss', sam_sup_loss.item(), iter_num)
            writer.add_scalar('train/sam_unsup_loss', sam_unsup_loss.item(), iter_num)
            writer.add_scalar('train/unet_sup_loss', unet_sup_loss.item(), iter_num)
            writer.add_scalar('train/unet_unsup_loss', unet_unsup_loss.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            if p_bar is not None:
                p_bar.update()

            # 进度条简要显示（按数据集自适应显示各分量）
            if args.dataset == 'fundus':
                p_bar.set_description('iter %d:L:%.4f,sSL:%.4f,sUL:%.4f,uSL:%.4f,uUL:%.4f,%.4f,%.4f,cons:%.4f,mask:%.4f,sd:%.4f,%.4f,ud:%.4f,%.4f,d:%.4f,%.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                           unet_size_mask.mean(), sam_ulb_dice[0], sam_ulb_dice[1], unet_ulb_dice[0], unet_ulb_dice[1], ulb_dice[0], ulb_dice[1], 
                                           np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                p_bar.set_description('iter %d: L:%.4f, sSL: %.4f, sUL: %.4f, uSL: %.4f, uUL: %.4f,%.4f,%.4f, cons: %.4f, mask: %.4f, sd: %.4f, ud: %.4f, d: %.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                           unet_size_mask.mean(), sam_ulb_dice[0], unet_ulb_dice[0], ulb_dice[0], 
                                           np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            elif args.dataset == 'MNMS':
                p_bar.set_description('iter %d:L:%.4f,sSL:%.4f,sUL:%.4f,uSL:%.4f,uUL:%.4f,%.4f,%.4f,cons:%.4f,mask:%.4f,sd:%.4f,%.4f,%.4f,ud:%.4f,%.4f,%.4f,d:%.4f,%.4f,%.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), sam_ulb_dice[0], sam_ulb_dice[1], sam_ulb_dice[2], unet_ulb_dice[0], unet_ulb_dice[1], unet_ulb_dice[2], ulb_dice[0], ulb_dice[1], ulb_dice[2],
                                        np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            # 每 num_eval_iter 次评估/打印统计量（与论文表/图风格对应）
            if iter_num % args.num_eval_iter == 0:
                if args.dataset == 'fundus':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,%.6f,ud:%.6f,%.6f,d:%.6f,%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, SAM_ulb_dice_sta[1].avg, unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg, ensemble_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[1].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,ud:%.6f,d:%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, unet_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[0].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                elif args.dataset == 'MNMS':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,%.6f,%.6f,ud:%.6f,%.6f,%.6f,d:%.6f,%.6f,%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, SAM_ulb_dice_sta[1].avg, SAM_ulb_dice_sta[2].avg, unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg, unet_ulb_dice_sta[2].avg, ensemble_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[1].avg, ensemble_ulb_dice_sta[2].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                text = ''
                for n, p in enumerate(part):
                    text += 'sam_ulb_%s_dice:%f' % (p, SAM_ulb_dice_sta[n].avg)
                    text += ', '
                for n, p in enumerate(part):
                    text += 'unet_ulb_%s_dice:%f' % (p, unet_ulb_dice_sta[n].avg)
                    text += ', '
                for n, p in enumerate(part):
                    text += 'ulb_%s_dice:%f' % (p, ensemble_ulb_dice_sta[n].avg)
                    if n != n_part-1:
                        text += ', '
                logging.info(text)

        if p_bar is not None:
            p_bar.close()

        # -------------------- 验证：先 U-Net 再 SAM（与论文表格分别报告两者性能一致） --------------------
        logging.info('test unet model')
        text = ''
        val_dice = test(args, unet_model, test_dataloader, epoch_num+1, writer, model_name='unet')
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
            save_text = "unet_avg_dice_best_model.pth"
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg unet model to {}'.format(save_best))
            if args.save_model:
                torch.save(unet_model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)

        logging.info('test sam model')
        stu_val_dice = test(args, SAM_model, test_dataloader, epoch_num+1, writer, model_name='SAM')
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "SAM_avg_dice_best_model.pth"
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg SAM model to {}'.format(save_best))
            if args.save_model:
                try:
                    SAM_model.save_lora_parameters(save_best)
                except:
                    SAM_model.module.save_lora_parameters(save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
    writer.close()


if __name__ == "__main__":
    # 结果保存目录按实验名/参数组合组织
    if len(args.save_name) == 0:
        args.save_name = f'fixmatch_{args.model}{args.img_size}_CP_lb{args.lb_num}_dm{args.lb_domain}'
    # snapshot_path = "../model/" + args.dataset + f"/{sys.argv[0].split('.')[0]}/" + args.save_name + "/"
    snapshot_path = os.path.join(args.output_root, args.dataset, f"{sys.argv[0].split('.')[0]}", args.save_name)
    snapshot_path += "/"  # 保持末尾斜杠
    
    # ----------------- 各数据集专属配置（论文 §4.1 数据与预处理）----------------- :contentReference[oaicite:26]{index=26}
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
        part = ['cup', 'disc']                 # 两个部位：视杯/视盘
        dataset = FundusSegmentation
        num_channels = 3
        patch_size = 256
        num_classes = 2
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = 30000
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
        num_channels = 1
        patch_size = 384
        num_classes = 1
        part = ['base'] 
        dataset = ProstateSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'MNMS':
        train_data_path="../../data/mnms"
        num_channels = 1
        patch_size = 288
        num_classes = 3
        part = ['lv', 'myo', 'rv'] 
        dataset = MNMSSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = 60000
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'BUSI':
        train_data_path="../../data/Dataset_BUSI_with_GT"
        num_channels = 1
        patch_size = 256
        num_classes = 1
        part = ['base'] 
        dataset = BUSISegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = 30000
        if args.domain_num >= 2:
            args.domain_num = 2
    
    # 小样本时收敛稳定性：缩小 batch
    if args.lb_num < 8:
        args.label_bs = 2
        args.unlabel_bs = 2
    else:
        args.label_bs = 4
        args.unlabel_bs = 4

    # SAM 是否多掩膜输出由类别数决定（多类 => multimask_output=True）
    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    n_part = len(part)
    # 指标计算函数表：二类/三类/单类的Dice定义不同
    dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff, 'MNMS':metrics.dice_coeff_3label, 'BUSI':metrics.dice_coeff}

    # 预训练权重：SAM or MedSAM（论文中默认使用 MedSAM 作为 Foundation）:contentReference[oaicite:27]{index=27}
    ckpt = {'SAM':'../../checkpoints/sam_vit_b_01ec64.pth', 'MedSAM':'../../checkpoints/medsam_vit_b.pth'}
    args.ckpt = ckpt[args.model]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 确定性：固定随机种子+禁用 cudnn.benchmark
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # 输出目录安全性控制
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./{}'.format(sys.argv[0]), snapshot_path + '/{}'.format(sys.argv[0]))

    # 日志：同时写文件与控制台
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    # 启动训练
    train(args, snapshot_path)
