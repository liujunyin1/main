# -*- coding: utf-8 -*-
import os, sys, argparse, time, json, math, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

# 本项目模块
from bridges.bridge_2d3d import slice_volume_to_tiles, tiles_to_volume
from networks.vnet_adapter import VNet3D
from smc.smc_3d import smc_fuse_3d
from regularizers.cdcr_3d import cdcr_loss_3d
from losses.losses_3d import ce_dice_3d
from utils.ema import update_ema
from utils.logging_utils import create_logger
from utils.checkpoint import save_ckpt, load_ckpt
from utils.metrics_3d import dice_per_class

# 复用（请从你的 SynFoC 复制过来）
# from segment_anything.build_sam import build_sam_vit_b as build_sam  # 你也可换其它SAM变体
from sam_lora_image_encoder import build_sam_vit_b_lora as build_sam  # 若你用LoRA版本，替换构造函数

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark, cudnn.deterministic = True, False

def get_args():
    p = argparse.ArgumentParser("SynFoC-3D Training")
    # 数据
    p.add_argument("--labeled_list", type=str, required=True,
                   help="标注数据索引(txt/csv)：每行 image_path,label_path")
    p.add_argument("--unlabeled_list", type=str, required=True,
                   help="无标注数据索引(txt)：每行 image_path")
    p.add_argument("--val_list", type=str, default=None, help="验证集索引(可选)")
    p.add_argument("--image_suffix", type=str, default=".nii.gz", help="数据后缀（.npy/.nii/.nii.gz）")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96,128,128])
    p.add_argument("--spacing", type=float, nargs=3, default=[1.5,1.0,1.0], help="resample spacing(mm), 若无需重采样可忽略")
    # 训练
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--train_bs", type=int, default=1)
    p.add_argument("--val_bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_m", type=float, default=0.999)
    p.add_argument("--lambda_u_max", type=float, default=1.0)
    p.add_argument("--cdcr_w", type=float, default=0.1)
    p.add_argument("--sam_T", type=float, default=1.5, help="SMC：SAM 温度标定")
    p.add_argument("--vnet_T", type=float, default=1.2, help="SMC：VNet 温度标定")
    p.add_argument("--smc_mode", type=str, default="voxel", choices=["voxel"])
    p.add_argument("--tile_cols", type=int, default=None, help="SAM大图网格列数，None自动开根")
    p.add_argument("--amp", action="store_true", help="开启混合精度")
    # 日志/断点
    p.add_argument("--outdir", type=str, default="./runs/syncf_3d")
    p.add_argument("--exp_name", type=str, default="default")
    p.add_argument("--resume", type=str, default="", help="从该 ckpt 路径恢复")
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # 设备
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()

def ramp_up(epoch, max_val, ramp_len=40):
    if ramp_len <= 0: return max_val
    if epoch >= ramp_len: return max_val
    # sigmoid ramp (同 SynFoC/MeanTeacher 常用)
    p = 1.0 - (epoch / ramp_len)
    return max_val * math.exp(-5.0 * p * p)

def build_dataloaders(args):
    from dataloaders.dataloader_3d import build_semi_loaders
    return build_semi_loaders(
        labeled_list=args.labeled_list,
        unlabeled_list=args.unlabeled_list,
        val_list=args.val_list,
        image_suffix=args.image_suffix,
        in_channels=args.in_channels,
        patch_size=tuple(args.patch_size),
        spacing=tuple(args.spacing),
        batch_size_l=args.train_bs,
        batch_size_u=args.train_bs,   # 这里简单设置一致
        batch_size_v=args.val_bs
    )

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    os.makedirs(os.path.join(args.outdir, args.exp_name), exist_ok=True)
    log_path = os.path.join(args.outdir, args.exp_name, "train.log")
    logger = create_logger(log_path)
    logger.info("Args:\n" + json.dumps(vars(args), indent=2))

    # ===== 数据 =====
    train_loader_l, train_loader_u, val_loader = build_dataloaders(args)
    logger.info(f"Labeled iters/epoch={len(train_loader_l)}, Unlabeled iters/epoch={len(train_loader_u)}")

    # ===== 模型（学生与教师）=====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SAM 2D
    sam_student = build_sam(checkpoint=None).to(device)
    sam_teacher = build_sam(checkpoint=None).to(device)
    for p in sam_teacher.parameters(): p.requires_grad = False
    # VNet 3D（SFR 的模型包装）
    vnet_student = VNet3D(in_ch=args.in_channels, out_ch=args.num_classes).to(device)
    vnet_teacher = VNet3D(in_ch=args.in_channels, out_ch=args.num_classes).to(device)
    for p in vnet_teacher.parameters(): p.requires_grad = False

    # 优化器
    params = list(vnet_student.parameters()) + [p for p in sam_student.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    start_epoch, best_dice = 0, 0.0
    ckpt_dir = os.path.join(args.outdir, args.exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ===== 恢复训练 =====
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading ckpt: {args.resume}")
        ckpt = load_ckpt(args.resume, map_location=device)
        sam_student.load_state_dict(ckpt["sam_student"])
        sam_teacher.load_state_dict(ckpt["sam_teacher"])
        vnet_student.load_state_dict(ckpt["vnet_student"])
        vnet_teacher.load_state_dict(ckpt["vnet_teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch={start_epoch}, best_dice={best_dice:.4f}")

    # ===== 训练 =====
    for epoch in range(start_epoch, args.epochs):
        sam_student.train(); vnet_student.train()
        t0 = time.time()
        lambda_u = ramp_up(epoch, max_val=args.lambda_u_max, ramp_len=40)
        epoch_loss, n_step = 0.0, 0

        # 将有标与无标 loader zip 到同一循环（按较小者长度）
        for (xb, yb), (uw, uc) in zip(train_loader_l, train_loader_u):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True).long()
            uw, uc = uw.to(device, non_blocking=True), uc.to(device, non_blocking=True)

            with autocast(enabled=args.amp):
                # ----- 教师在 Uw（弱增）上预测 -----
                # SAM 教师：Uw 3D→2D 大图
                uw_big, meta_u = slice_volume_to_tiles(uw, tile_cols=args.tile_cols)
                with torch.no_grad():
                    samT_logits_2d = sam_teacher(uw_big)             # 假定返回 logits [B,C,H*,W*]
                samT_logits_3d = tiles_to_volume(samT_logits_2d, meta_u)

                # VNet 教师：Uw 3D
                with torch.no_grad():
                    vnetT_logits_3d = vnet_teacher(uw)

                # ----- SMC 融合（3D 域）-----
                p_en_3d, alpha = smc_fuse_3d(
                    logits_vnet=vnetT_logits_3d,
                    logits_sam3d=samT_logits_3d,
                    T_v=args.vnet_T, T_s=args.sam_T, mode=args.smc_mode
                )
                pseudo_argmax = p_en_3d.argmax(1).detach()

                # ----- 学生在 Uc（无标-强/中增）-----
                uc_big, meta_c = slice_volume_to_tiles(uc, tile_cols=args.tile_cols)
                samS_logits_2d = sam_student(uc_big)
                samS_logits_3d = tiles_to_volume(samS_logits_2d, meta_c)
                vnetS_logits_3d = vnet_student(uc)

                # 无监督
                Lu_sam  = ce_dice_3d(samS_logits_3d, pseudo_argmax)
                Lu_vnet = ce_dice_3d(vnetS_logits_3d, pseudo_argmax)

                # CDCR（3D 同域）：边界梯度一致 + 分布一致
                ps3d = torch.softmax(samS_logits_3d, dim=1)
                pv3d = torch.softmax(vnetS_logits_3d, dim=1)
                L_cdcr = cdcr_loss_3d(ps3d, pv3d)

                # ----- 有标弱增监督（Xw） -----
                x_big, meta_x = slice_volume_to_tiles(xb, tile_cols=args.tile_cols)
                sam_logits_x_2d = sam_student(x_big)
                sam_logits_x_3d = tiles_to_volume(sam_logits_x_2d, meta_x)
                vnet_logits_x_3d = vnet_student(xb)
                Lx_sam  = ce_dice_3d(sam_logits_x_3d, yb)
                Lx_vnet = ce_dice_3d(vnet_logits_x_3d, yb)

                loss = (Lx_sam + Lx_vnet) + lambda_u*(Lu_sam + Lu_vnet) + args.cdcr_w*L_cdcr

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA
            update_ema(sam_teacher, sam_student, ema_m=args.ema_m)
            update_ema(vnet_teacher, vnet_student, ema_m=args.ema_m)

            epoch_loss += loss.item(); n_step += 1

        # ===== 验证（可选）=====
        val_dice = 0.0
        if val_loader is not None:
            sam_student.eval(); vnet_student.eval()
            dices = []
            with torch.no_grad():
                for (xv, yv) in val_loader:
                    xv, yv = xv.to(device), yv.to(device).long()
                    # 直接用 VNet 学生评估；也可做 SAM 回拼评估
                    logits = vnet_student(xv)
                    prob = torch.softmax(logits, dim=1)
                    dices.append(dice_per_class(prob, yv, num_classes=args.num_classes).cpu().numpy())
            if len(dices)>0:
                val_dice = float(np.mean(dices))
                best_dice = max(best_dice, val_dice)

        # ===== 日志 & 保存 =====
        dt = time.time()-t0
        logger.info(f"Epoch {epoch:03d} | loss={epoch_loss/max(1,n_step):.4f} | val_dice={val_dice:.4f} | "
                    f"lambda_u={lambda_u:.3f} | time={dt:.1f}s")

        if (epoch % args.save_interval == 0) or (epoch==args.epochs-1):
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
            save_ckpt(ckpt_path,
                      epoch=epoch, best_dice=best_dice,
                      sam_student=sam_student.state_dict(),
                      sam_teacher=sam_teacher.state_dict(),
                      vnet_student=vnet_student.state_dict(),
                      vnet_teacher=vnet_teacher.state_dict(),
                      optimizer=optimizer.state_dict(),
                      scaler=scaler.state_dict())
            logger.info(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()

