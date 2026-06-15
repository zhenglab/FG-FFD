import os
import cv2
import numpy as np
import csv
from glob import glob
import models.custom_model as custom_model
import re



def get_demo_data(demo_dir, num_segments=1):
    flow_dir = os.path.join(demo_dir, 'flow')
    frame_dir = os.path.join(demo_dir, 'frame')
    def extract_frame_id(path_str):
        numbers = re.findall(r'\d+', path_str)
        return int(numbers[-1]) if numbers else 0
    rgb_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if os.path.isfile(os.path.join(frame_dir, f))], key=extract_frame_id)
    def extract_flow_key(path_str):
        numbers = re.findall(r'\d+', path_str)
        frame_id = int(numbers[-1]) if numbers else 0
        axis_weight = 0 if 'x' in os.path.basename(path_str).lower() else 1
        return (frame_id, axis_weight)
    flow_paths = sorted([os.path.join(flow_dir, f) for f in os.listdir(flow_dir) if os.path.isfile(os.path.join(flow_dir, f))], key=extract_flow_key)

    valid_range = len(rgb_paths) - 4
    if valid_range <= 0:
        start_indices = [0] * num_segments
    else:
        start_indices = np.linspace(0, valid_range, num_segments).astype(int)

    segment_rgb_list = []
    segment_flow_list = []

    for start_idx in start_indices:
        current_rgb_paths = rgb_paths[start_idx : start_idx + 4]
        rgb_frames = []
        for path in current_rgb_paths:
            img = Image.open(path).convert('RGB') # 确保3通道
            img_arr = np.asarray(img)             # Shape: (H, W, C)
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
            rgb_frames.append(img_tensor)
        processed_rgb = torch.stack(rgb_frames)
        segment_rgb_list.append(processed_rgb)
        flow_start = start_idx * 2 - 6 
        flow_end = start_idx * 2 + 14
        
        if flow_start < 0:
            flow_start, flow_end = 0, 20
        elif flow_end > len(flow_paths):
            flow_start = max(0, len(flow_paths) - 20)
            flow_end = len(flow_paths)
            
        flow_segment_paths = flow_paths[flow_start:flow_end]
        
        flow_tensors = []
        for path in flow_segment_paths:
            flow_img = Image.open(path)
            flow_arr = np.asarray(flow_img) # 假设单通道或双通道光流图
            flow_tensors.append(torch.from_numpy(flow_arr).float())
        
        current_flow_fea = torch.stack(flow_tensors) 
        flow_1 = current_flow_fea[0:10].unsqueeze(0)
        flow_2 = current_flow_fea[2:12].unsqueeze(0)
        flow_3 = current_flow_fea[4:14].unsqueeze(0)
        flow_4 = current_flow_fea[6:16].unsqueeze(0)
        flow_5 = current_flow_fea[8:18].unsqueeze(0)
        flow_6 = current_flow_fea[10:20].unsqueeze(0)
        
        flow_fea = torch.cat([flow_1, flow_2, flow_3, flow_4, flow_5, flow_6], dim=0)
        segment_flow_list.append(flow_fea)

    final_process_imgs = torch.stack(segment_rgb_list) # Shape: (num_segments, 4, C, H, W)
    final_flow_fea = torch.stack(segment_flow_list)     # Shape: (num_segments, 6, 10, H, W)

    return final_process_imgs, final_flow_fea


def get_model():
    model = custom_model.FinalModel_v1(pretrained=False)
    ckpt_load_path = 'checkpoints/Stage2_FinalModel_v1/ckpt/Final.tar'
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint
    new_state_dict = {}    
    for k, v in sd.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
            new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict,strict=False)
    print('sdload', msg)

    return model


import torch
from PIL import Image
import sys
if __name__ == "__main__":

    demo_path = "./demo/id0_id1_0005"
    rgb_tensor, flow_tensor = get_demo_data(demo_path, num_segments=8)
    print("RGB Shape:", rgb_tensor.shape)   # 应该是 [1, 4, 3, H, W]
    print("Flow Shape:", flow_tensor.shape) # 应该是 [1, 6, 10, H, W]
    model = get_model()
    with torch.no_grad():
        output = model(flow_tensor.unsqueeze(0), rgb_tensor.unsqueeze(0), is_eval=True)
    outputs=torch.nn.functional.softmax(output, dim=2)[:,:,1]  # [B, N*T, 2]
    print(outputs)

