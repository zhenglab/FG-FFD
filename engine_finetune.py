import os
import sys
import time
import torch
from utils import *
from common.utils import *
import glob as glob_lib
from models import *
from datasets import *
from tqdm import tqdm
from collections import OrderedDict
from common.utils import map_util
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
torch.autograd.set_detect_anomaly(True)
import torchvision

def train_one_epoch(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        labels = datas['labels']
        # video_path = datas['video_path']
        # sampled_frame_idxs = datas['index']

        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)

        outputs= model(flow_fea, images)
        loss_sim = torch.tensor(0.0).cuda()
        if isinstance(outputs, list):
            # outputs = outputs[0] + outputs[1].unsqueeze(2).repeat(1,1,2,1).flatten(1,2)+ outputs[2].unsqueeze(2).repeat(1,1,4,1).flatten(1,2)+ outputs[3].unsqueeze(2).repeat(1,1,8,1).flatten(1,2)+ outputs[4].unsqueeze(2).repeat(1,1,16,1).flatten(1,2)+ outputs[5].unsqueeze(2).repeat(1,1,32,1).flatten(1,2)
            target_vals = (1 - labels).float() 
            target_vals = target_vals.view(-1, 1, 1, 1).expand_as(outputs[1])
            loss_sim = F.binary_cross_entropy_with_logits(outputs[1], target_vals)
            labels_r = labels.unsqueeze(1).repeat(1,outputs[0].size(1)).flatten(0,1)
            outputs_r2 = outputs[0].flatten(0,1)
            outputs = outputs[0]
        else:
            outputs, loss_sim = outputs
            labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
            outputs_r2 = outputs.flatten(0,1)
        
        loss = criterion(outputs_r2, labels_r)+loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%500==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                input_real = images[labels == 0]
                input_fake = images[labels == 1]

                outputs_real = outputs[labels == 0].squeeze(1)
                outputs_fake = outputs[labels == 1].squeeze(1)                    
                # images = map_util.save_real_data_images_withrebuild(video_path,outputs_real, outputs_fake, input_real, input_fake, input_real, input_fake, labels, args.transform_params.mean, args.transform_params.std, sampled_frame_idxs,img_save_dir, str(epoch)+"-"+str(idx), return_np=False)
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()


def train_one_epoch2loss(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses1 = AverageMeter('Loss1', ':.4f')
    losses2 = AverageMeter('Loss2', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses1, losses2, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        labels = datas['labels']
        # video_path = datas['video_path']
        # sampled_frame_idxs = datas['index']

        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)

        outputs= model(flow_fea, images)
        loss_sim = torch.tensor(0.0).cuda()
        if isinstance(outputs, list):
            # outputs = outputs[0] + outputs[1].unsqueeze(2).repeat(1,1,2,1).flatten(1,2)+ outputs[2].unsqueeze(2).repeat(1,1,4,1).flatten(1,2)+ outputs[3].unsqueeze(2).repeat(1,1,8,1).flatten(1,2)+ outputs[4].unsqueeze(2).repeat(1,1,16,1).flatten(1,2)+ outputs[5].unsqueeze(2).repeat(1,1,32,1).flatten(1,2)
            target_vals = (1 - labels).float() 
            target_vals = target_vals.view(-1, 1, 1, 1).expand_as(outputs[1])
            loss_sim = F.binary_cross_entropy_with_logits(outputs[1], target_vals)
            labels_r = labels.unsqueeze(1).repeat(1,outputs[0].size(1)).flatten(0,1)
            outputs_r2 = outputs[0].flatten(0,1)
            outputs = outputs[0]
        else:
            outputs, outputs2,loss_sim = outputs
            labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
            outputs_r2 = outputs.flatten(0,1)
            outputs_r22 = outputs2.flatten(0,1)

        loss1 = criterion(outputs_r2, labels_r)
        loss2 = criterion(outputs_r22, labels_r)
        loss = loss1+loss2+loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%500==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                input_real = images[labels == 0]
                input_fake = images[labels == 1]

                outputs_real = outputs[labels == 0].squeeze(1)
                outputs_fake = outputs[labels == 1].squeeze(1)                    
                # images = map_util.save_real_data_images_withrebuild(video_path,outputs_real, outputs_fake, input_real, input_fake, input_real, input_fake, labels, args.transform_params.mean, args.transform_params.std, sampled_frame_idxs,img_save_dir, str(epoch)+"-"+str(idx), return_np=False)
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_mask_with_patches(masks_tensor, video_idx=0, frame_idx=0, save_path="mask_visualization.png"):
    """
    参数:
        masks_tensor: torch.Tensor, shape [16, 4, 224, 224, 1]
        video_idx: int, 选择第几个视频 (0-15)
        frame_idx: int, 选择第几帧 (0-3)
        save_path: str, 保存图片的路径
    """
    # 1. 数据提取与预处理
    # 取出指定视频和帧的 Mask: [224, 224, 1] -> [224, 224]
    mask = masks_tensor[video_idx, frame_idx].squeeze().detach().cpu()
    
    # 2. 计算 14x14 个块的均值
    # 图像尺寸 224x224, 目标 Grid 14x14 => Patch Size = 16x16
    patch_size = 16 
    
    # 使用 AvgPool2d 快速计算每个 16x16 窗口内的均值
    # 输入需要是 BCHW 格式，所以先 unsqueeze 增加 Batch 和 Channel 维度
    # input: [1, 1, 224, 224] -> output: [1, 1, 14, 14]
    mask_input = mask.unsqueeze(0).unsqueeze(0)
    patch_means = F.avg_pool2d(mask_input, kernel_size=patch_size, stride=patch_size)
    
    # 移除多余维度 -> [14, 14]
    patch_means_np = patch_means.squeeze().numpy()
    mask_np = mask.numpy()

    # 3. 可视化绘图
    plt.figure(figsize=(12, 12), dpi=100) # 设置大一点的图，防止文字重叠
    
    # 显示原始 Mask
    # 使用 gray colormap，并在上面叠加信息
    plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1, aspect='equal')
    
    # 绘制网格线
    height, width = mask_np.shape
    num_h = patch_means_np.shape[0] # 14
    num_w = patch_means_np.shape[1] # 14
    
    # 画竖线
    for x in range(0, width + 1, patch_size):
        plt.axvline(x - 0.5, color='red', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 画横线
    for y in range(0, height + 1, patch_size):
        plt.axhline(y - 0.5, color='red', linestyle='-', linewidth=0.5, alpha=0.5)

    # 4. 在每个格子里填写均值
    for i in range(num_h): # 14 行
        for j in range(num_w): # 14 列
            # 获取均值
            val = patch_means_np[i, j]
            
            # 计算文字显示的中心坐标
            # i 是行索引（对应 y 轴），j 是列索引（对应 x 轴）
            center_x = j * patch_size + patch_size / 2
            center_y = i * patch_size + patch_size / 2
            
            # 为了可读性：如果 mask 很亮（接近1），文字用红色/黑色；如果 mask 很暗，文字用黄色/白色
            # 这里简单起见，使用鲜艳的颜色（如青色或红色）
            # 格式化保留两位小数
            text_val = f"{val:.2f}"
            
            # 只有当数值不为0时才显示，或者全部显示（根据需求调整）
            # 这里稍微调小字体，防止 14x14 太挤
            if val > 0.01: # 仅为了看起来清爽一点，可去掉这个if显示所有
                plt.text(center_x, center_y, text_val, 
                         color='cyan', ha='center', va='center', 
                         fontsize=7, fontweight='bold')
            else:
                # 0值用较淡的颜色显示
                plt.text(center_x, center_y, "0", 
                         color='yellow', ha='center', va='center', 
                         fontsize=6, alpha=0.5)

    plt.title(f"Mask Visualization (Video {video_idx}, Frame {frame_idx})\nGrid: 14x14, Patch Mean Values", fontsize=14)
    plt.axis('off') # 关闭坐标轴刻度
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path)
    print(f"图像已保存至: {save_path}")

def generate_patch_labels(masks, grid_size=14, threshold=0.0):
    """
    输入 masks: [B, T, H, W, C] 或 [B, T, H, W]
    输出 labels: [B, T, 196]
    """
    # 1. 维度处理
    # 假设输入是 [16, 4, 224, 224, 1]
    if masks.dim() == 5:
        B, T, H, W, C = masks.shape
        # 将 Batch 和 Time 维度合并，方便并行处理 -> [B*T, C, H, W]
        masks_reshaped = masks.view(B * T, C, H, W)
        # 还要确保通道维度在第二位，如果输入是 [B, T, H, W, 1]，view后就是 [N, 1, 224, 224]
        # 如果是 [B, T, H, W, 1] -> permute -> [B, T, 1, H, W] 可能更安全，但这里view够用了
        masks_reshaped = masks.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    else:
        # 如果没有通道维度 [16, 4, 224, 224]
        B, T, H, W = masks.shape
        masks_reshaped = masks.view(B * T, 1, H, W)

    # 2. 计算 Patch Size
    # 224 / 14 = 16
    patch_size = H // grid_size
    
    # 3. 下采样 (Downsample) 得到 14x14
    # 使用 AvgPool2d，结果表示该 Patch 内 mask 的平均值（覆盖率）
    # 输出 shape: [B*T, 1, 14, 14]
    downsampled_map = F.avg_pool2d(masks_reshaped, kernel_size=patch_size, stride=patch_size)
    
    # 4. 生成 Label (二值化)
    # 逻辑：如果该块内的平均值 > threshold (0)，则认为该块属于 mask (label=1)
    # float() 将 True/False 转为 1.0/0.0
    patch_labels = (downsampled_map > threshold).float()
    
    # 5. 拉平为 196
    # [B*T, 1, 14, 14] -> [B*T, 196]
    patch_labels_flat = patch_labels.flatten(start_dim=1)
    
    # 6. 恢复 [B, T, 196] 结构
    final_labels = patch_labels_flat.view(B, T, -1)
    
    return final_labels, downsampled_map



def train_one_epoch1loss(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses1 = AverageMeter('Loss1', ':.4f')
    losses2 = AverageMeter('Loss2', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses1, losses2, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        images_ori = datas['process_imgs_ori']
        images_aug = datas['process_imgs_aug']
        flow_ori = datas['flow_fea_ori']
        flow_aug = datas['flow_fea']
        labels = datas['labels']
        masks = datas['mask']
        # visualize_mask_with_patches(masks, video_idx=0, frame_idx=2, save_path="mask_patch_view.png")
        maskslabels, downsampled_map = generate_patch_labels(masks, grid_size=14, threshold=0.0)
        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)

        images_ori = images_ori.cuda(args.local_rank)
        images_aug = images_aug.cuda(args.local_rank)
        flow_ori = flow_ori.cuda(args.local_rank)
        flow_aug = flow_aug.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs= model(flow_ori, images_ori, flow_aug, images_aug, maskslabels)
        

        outputs, outputs2, loss_sim = outputs
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        # labels_r2 = torch.ones_like(labels_r)
        outputs_r2 = outputs.flatten(0,1)
        # outputs_r22 = outputs2.flatten(0,1)
        

        loss1 = criterion(outputs_r2, labels_r)
        # loss2 = criterion(outputs_r22, labels_r2)
        loss = loss1
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses1.update(loss1.item(), images.size(0))
        # losses2.update(loss2.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%100==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                os.makedirs(img_save_dir, exist_ok=True)

                B = images_ori.size(0)
                H, W = images_ori.size(-2), images_ori.size(-1)

                # 原图/增广图: [B, 1, 1, 3, H, W] -> [B, 3, H, W]
                # 用 view 去掉多余的维度
                img_ori_4d = images_ori.view(B, 3, H, W)
                img_aug_4d = images_aug.view(B, 3, H, W)

                # Mask: [B, 1, H, W, 1] -> 转换为 [B, 1, H, W]
                # 先用 squeeze(1) 去掉第2维(size=1)，变成 [B, H, W, 1]
                # 再用 permute 把它变成 PyTorch 通道在前的格式 [B, 1, H, W]
                mask_4d = masks.squeeze(1).permute(0, 3, 1, 2)
                # ===============================================

                n_vis = min(8, B)
                v_ori = img_ori_4d[:n_vis].cpu()
                v_aug = img_aug_4d[:n_vis].cpu()
                v_mask = mask_4d[:n_vis].cpu()

                # ================= 2. 反归一化 =================
                mean = torch.tensor(args.transform_params.mean).view(1, 3, 1, 1)
                std = torch.tensor(args.transform_params.std).view(1, 3, 1, 1)
                v_ori = v_ori * std + mean
                v_aug = v_aug * std + mean

                # ================= 3. Mask 通道对齐 =================
                # 把单通道 [B, 1, H, W] 的 mask 复制成三通道 [B, 3, H, W]，以便和原图拼接
                if v_mask.size(1) == 1:
                    v_mask = v_mask.repeat(1, 3, 1, 1) 

                # ================= 4. 防止溢出 =================
                v_ori = torch.clamp(v_ori, 0, 1)
                v_aug = torch.clamp(v_aug, 0, 1)
                v_mask = torch.clamp(v_mask, 0, 1)

                # ================= 5. 交错拼接 =================
                vis_list = []
                for i in range(n_vis):
                    vis_list.append(v_ori[i:i+1])
                    vis_list.append(v_aug[i:i+1])
                    vis_list.append(v_mask[i:i+1])
                
                vis_tensor = torch.cat(vis_list, dim=0)
                
                # 保存为网格图: 每行 3 张图 (原图 -> 增广图 -> Mask)
                save_path = os.path.join(img_save_dir, f"vis_epoch{epoch}_idx{idx}.png")
                torchvision.utils.save_image(vis_tensor, save_path, nrow=3, normalize=False)
                # =======================================================================        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()


def train_one_epoch2loss_SBI(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses1 = AverageMeter('Loss1', ':.4f')
    losses2 = AverageMeter('Loss2', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses1, losses2, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        images_ori = datas['process_imgs_ori']
        images_aug = datas['process_imgs_aug']
        flow_ori = datas['flow_fea_ori']
        flow_aug = datas['flow_fea']
        labels = datas['labels']
        masks = datas['mask']
        # visualize_mask_with_patches(masks, video_idx=0, frame_idx=2, save_path="mask_patch_view.png")
        maskslabels, downsampled_map = generate_patch_labels(masks, grid_size=14, threshold=0.0)
        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)

        images_ori = images_ori.cuda(args.local_rank)
        images_aug = images_aug.cuda(args.local_rank)
        flow_ori = flow_ori.cuda(args.local_rank)
        flow_aug = flow_aug.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs= model(flow_ori, images_ori, flow_aug, images_aug, maskslabels)
        

        outputs, outputs2, loss_sim = outputs
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        labels_r2 = torch.ones_like(labels_r)
        outputs_r2 = outputs.flatten(0,1)
        outputs_r22 = outputs2.flatten(0,1)
        
        loss1 = criterion(outputs_r2, labels_r)
        loss2 = criterion(outputs_r22, labels_r2)
        loss = loss1+loss2+loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%100==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                os.makedirs(img_save_dir, exist_ok=True)

                B = images_ori.size(0)
                H, W = images_ori.size(-2), images_ori.size(-1)

                # 原图/增广图: [B, 1, 1, 3, H, W] -> [B, 3, H, W]
                # 用 view 去掉多余的维度
                img_ori_4d = images_ori.view(B, 3, H, W)
                img_aug_4d = images_aug.view(B, 3, H, W)

                # Mask: [B, 1, H, W, 1] -> 转换为 [B, 1, H, W]
                # 先用 squeeze(1) 去掉第2维(size=1)，变成 [B, H, W, 1]
                # 再用 permute 把它变成 PyTorch 通道在前的格式 [B, 1, H, W]
                mask_4d = masks.squeeze(1).permute(0, 3, 1, 2)
                # ===============================================

                n_vis = min(8, B)
                v_ori = img_ori_4d[:n_vis].cpu()
                v_aug = img_aug_4d[:n_vis].cpu()
                v_mask = mask_4d[:n_vis].cpu()

                # ================= 2. 反归一化 =================
                mean = torch.tensor(args.transform_params.mean).view(1, 3, 1, 1)
                std = torch.tensor(args.transform_params.std).view(1, 3, 1, 1)
                v_ori = v_ori * std + mean
                v_aug = v_aug * std + mean

                # ================= 3. Mask 通道对齐 =================
                # 把单通道 [B, 1, H, W] 的 mask 复制成三通道 [B, 3, H, W]，以便和原图拼接
                if v_mask.size(1) == 1:
                    v_mask = v_mask.repeat(1, 3, 1, 1) 

                # ================= 4. 防止溢出 =================
                v_ori = torch.clamp(v_ori, 0, 1)
                v_aug = torch.clamp(v_aug, 0, 1)
                v_mask = torch.clamp(v_mask, 0, 1)

                # ================= 5. 交错拼接 =================
                vis_list = []
                for i in range(n_vis):
                    vis_list.append(v_ori[i:i+1])
                    vis_list.append(v_aug[i:i+1])
                    vis_list.append(v_mask[i:i+1])
                
                vis_tensor = torch.cat(vis_list, dim=0)
                
                # 保存为网格图: 每行 3 张图 (原图 -> 增广图 -> Mask)
                save_path = os.path.join(img_save_dir, f"vis_epoch{epoch}_idx{idx}.png")
                torchvision.utils.save_image(vis_tensor, save_path, nrow=3, normalize=False)
                # =======================================================================        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()

def train_one_epoch_s2(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses1 = AverageMeter('Loss1', ':.4f')
    losses2 = AverageMeter('Loss2', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses1, losses2, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        # images_ori = datas['process_imgs_ori']
        # images_aug = datas['process_imgs_aug']
        # flow_ori = datas['flow_fea_ori']
        # flow_aug = datas['flow_fea']
        labels = datas['labels']
        # masks = datas['mask']
        # visualize_mask_with_patches(masks, video_idx=0, frame_idx=2, save_path="mask_patch_view.png")
        # maskslabels, downsampled_map = generate_patch_labels(masks, grid_size=14, threshold=0.0)
        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)

        # images_ori = images_ori.cuda(args.local_rank)
        # images_aug = images_aug.cuda(args.local_rank)
        # flow_ori = flow_ori.cuda(args.local_rank)
        # flow_aug = flow_aug.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs= model(flow_fea, images,label=labels)
        

        outputs, loss_sim = outputs
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        # labels_r2 = torch.ones_like(labels_r)
        outputs_r2 = outputs.flatten(0,1)
        # outputs_r22 = outputs2.flatten(0,1)
        

        loss1 = criterion(outputs_r2, labels_r)
        loss = loss1+loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses1.update(loss1.item(), images.size(0))
        # losses2.update(loss2.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%500==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                input_real = images[labels == 0]
                input_fake = images[labels == 1]

                outputs_real = outputs[labels == 0].squeeze(1)
                outputs_fake = outputs[labels == 1].squeeze(1)                    
                # images = map_util.save_real_data_images_withrebuild(video_path,outputs_real, outputs_fake, input_real, input_fake, input_real, input_fake, labels, args.transform_params.mean, args.transform_params.std, sampled_frame_idxs,img_save_dir, str(epoch)+"-"+str(idx), return_np=False)
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()

def train_one_epoch_s2_2loss(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses1 = AverageMeter('Loss1', ':.4f')
    losses2 = AverageMeter('Loss2', ':.4f')
    losses_sim = AverageMeter('Loss_sim', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses1, losses2, losses_sim, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        flow_fea = datas['flow_fea']
        images = datas['process_imgs']
        # images_ori = datas['process_imgs_ori']
        # images_aug = datas['process_imgs_aug']
        # flow_ori = datas['flow_fea_ori']
        # flow_aug = datas['flow_fea']
        labels = datas['labels']
        # masks = datas['mask']
        # visualize_mask_with_patches(masks, video_idx=0, frame_idx=2, save_path="mask_patch_view.png")
        # maskslabels, downsampled_map = generate_patch_labels(masks, grid_size=14, threshold=0.0)
        images = images.cuda(args.local_rank)
        flow_fea = flow_fea.cuda(args.local_rank)

        # images_ori = images_ori.cuda(args.local_rank)
        # images_aug = images_aug.cuda(args.local_rank)
        # flow_ori = flow_ori.cuda(args.local_rank)
        # flow_aug = flow_aug.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs= model(flow_fea, images,label=labels)
        

        outputs, outputs2, loss_sim = outputs
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        # labels_r2 = torch.ones_like(labels_r)
        outputs_r2 = outputs.flatten(0,1)
        outputs_r22 = outputs2.flatten(0,1)
        # outputs_r22 = outputs2.flatten(0,1)
        

        loss1 = criterion(outputs_r2, labels_r)
        loss2 = criterion(outputs_r22, labels_r)
        loss = loss1+loss2+loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))
        losses_sim.update(loss_sim.item(), images.size(0))
        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()



from myeval import test,test_perepoch
from datasets.factory import get_final_dataloader

def test_one_epoch(model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    model.eval()
    if epoch == args.train.max_epoches:
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = global_step
        checkpoint['args'] = args
        checkpoint_save_name = "Final.tar".format(epoch, global_step)
        checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
        torch.save(checkpoint, checkpoint_save_dir)

    if epoch == args.train.max_epoches:
        with torch.no_grad():
            args.final_test.dataset.params.num_segments = 32
            args.final_test.dataset.params.method = 'DFDC'
            args.final_test.dataset.params.split = 'test'
            test_dataloader = get_final_dataloader(args, 'test')
            auc_dfdc, auc_dfdc_f = test_perepoch(test_dataloader, model, args)

            args.final_test.dataset.params.method = 'DFD'
            args.final_test.dataset.params.split = 'test'
            test_dataloader = get_final_dataloader(args, 'test')
            auc_dfd, auc_dfd_f = test_perepoch(test_dataloader, model, args)

            args.final_test.dataset.params.method = 'Celeb-DF'
            args.final_test.dataset.params.split = 'test'
            test_dataloader = get_final_dataloader(args, 'test')
            auc_cdf, auc_cdf_f = test_perepoch(test_dataloader, model, args)

            args.final_test.dataset.params.method = 'FFIW'
            args.final_test.dataset.params.split = 'test'
            test_dataloader = get_final_dataloader(args, 'test')
            auc_ffiw, auc_ffiw_f = test_perepoch(test_dataloader, model, args)
            auc_cdf = float(auc_cdf)
            auc_cdf_f = float(auc_cdf_f)
            auc_dfdc = float(auc_dfdc)
            auc_dfdc_f = float(auc_dfdc_f)
            auc_ffiw = float(auc_ffiw)
            auc_ffiw_f = float(auc_ffiw_f)
            auc_dfd = float(auc_dfd)
            auc_dfd_f = float(auc_dfd_f)
            test_info = '[TEST] Video/Frame: EPOCH-{} Celeb-DF_AUC: {:.4f} {:.4f} DFDC_AUC: {:.4f} {:.4f} FFIW_AUC: {:.4f} {:.4f} DFD_AUC: {:.4f} {:.4f}'.format(epoch,auc_cdf,auc_cdf_f,auc_dfdc,auc_dfdc_f,auc_ffiw,auc_ffiw_f,auc_dfd,auc_dfd_f)
            ckpt_path = os.path.join(args.exam_dir, 'ckpt')
            with open(ckpt_path+'/test_info.txt','a+') as file:
                file.write(test_info)
                file.write('\n')

        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1)



def test_one_epoch_s1(model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    model.eval()
    if epoch == args.train.max_epoches:
        if epoch == args.train.max_epoches:
            checkpoint = OrderedDict()
            checkpoint['state_dict'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['global_step'] = global_step
            checkpoint['args'] = args
            checkpoint_save_name = "Final.tar".format(epoch, global_step)
            checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
            torch.save(checkpoint, checkpoint_save_dir)


        with torch.no_grad():
            if epoch == args.train.max_epoches:
                args.final_test.dataset.params.num_segments = 32
                args.final_test.dataset.params.method = 'DFDC'
                args.final_test.dataset.params.split = 'test'
                test_dataloader = get_final_dataloader(args, 'test')
                auc_dfdc, auc_dfdc_f = test_perepoch(test_dataloader, model, args)

                args.final_test.dataset.params.method = 'DFD'
                args.final_test.dataset.params.split = 'test'
                test_dataloader = get_final_dataloader(args, 'test')
                auc_dfd, auc_dfd_f = test_perepoch(test_dataloader, model, args)

                args.final_test.dataset.params.method = 'Celeb-DF'
                args.final_test.dataset.params.split = 'test'
                test_dataloader = get_final_dataloader(args, 'test')
                auc_cdf, auc_cdf_f = test_perepoch(test_dataloader, model, args)

                args.final_test.dataset.params.method = 'FFIW'
                args.final_test.dataset.params.split = 'test'
                test_dataloader = get_final_dataloader(args, 'test')
                auc_ffiw, auc_ffiw_f = test_perepoch(test_dataloader, model, args)




                auc_cdf = float(auc_cdf)
                auc_cdf_f = float(auc_cdf_f)
                auc_dfdc = float(auc_dfdc)
                auc_dfdc_f = float(auc_dfdc_f)
                auc_ffiw = float(auc_ffiw)
                auc_ffiw_f = float(auc_ffiw_f)
                auc_dfd = float(auc_dfd)
                auc_dfd_f = float(auc_dfd_f)

                test_info = '[TEST] Video/Frame: EPOCH-{} Celeb-DF_AUC: {:.4f} {:.4f} DFDC_AUC: {:.4f} {:.4f} FFIW_AUC: {:.4f} {:.4f} DFD_AUC: {:.4f} {:.4f}'.format(epoch,auc_cdf,auc_cdf_f,auc_dfdc,auc_dfdc_f,auc_ffiw,auc_ffiw_f,auc_dfd,auc_dfd_f)
                ckpt_path = os.path.join(args.exam_dir, 'ckpt')
                with open(ckpt_path+'/test_info.txt','a+') as file:
                    file.write(test_info)
                    file.write('\n')
            if lr_scheduler is not None:
                lr_scheduler.step(epoch-1)
