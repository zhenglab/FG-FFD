import os
import numpy as np
import torch
import torch.utils.data as data
import random
from collections import OrderedDict
from common.utils import map_util
from PIL import Image
from .utils import dataloader_util
from tqdm import tqdm
import re
from einops import rearrange
import pickle
class Image_dataset_random_rgbwflow_s1_test(data.Dataset):
    def __init__(self,
                 root,
                 method='FaceForensics_c23',
                 split='train',
                 num_segments=8,
                 num_segments_flow=8,
                 transform=None,
                 cutout=False,
                 is_sbi=False,
                 image_size=224,
                 methods=None):
        super().__init__()#实例化
        self.root = root
        self.dataset_info = []
        self.method = method
        self.split = split
        self.num_segments = num_segments
        self.num_segments_flow = num_segments_flow
        self.transform = transform
        self.image_size = image_size
        self.is_cutout = cutout
        self.is_sbi = is_sbi
        self.cutout = map_util.Cutout()
        self.parse_dataset_info()
        print(self.split,self.transform)

    def sort_image_paths_by_frame_id(self, path_list):
        def extract_frame_id(path):
            path_str = str(path)
            numbers = re.findall(r'\d+', path_str)
            if not numbers:
                return 0
            return int(numbers[-1])
        sorted_list = sorted(path_list, key=extract_frame_id)
        return sorted_list

    def parse_dataset_info(self):#数据集解析
        if self.method== 'FaceForensics_c23':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    if 'original_sequences' in item[0]:
                        vid_id = item[0].split('/')[-1]
                        flow_path = f'/data4-16T/liuyingjie/datasets/FFD_flows/flow_processed_unified/original_sequences/{vid_id}'
                        flow_list=[]
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                flow_list.append(file_full_path)
                        if self.split =='train':
                            real_full_path = f'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-a/train/FF++/original_sequences/youtube/c23/videos/{vid_id}/frame'
                        else:
                            real_full_path = f'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-at/test/FF++/original_sequences/youtube/c23/videos/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(real_full_path):
                            file_full_path = os.path.join(real_full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        if os.path.exists(item[0]) and os.path.exists(real_full_path):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                            self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                    else:
                        fake_method = item[0].split('/')[-2]
                        vid_id = item[0].split('/')[-1]
                        flow_path = f'/data4-16T/liuyingjie/datasets/FFD_flows/flow_processed_unified/{fake_method}/{vid_id}'
                        flow_list=[]
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                flow_list.append(file_full_path)
                        if self.split == 'train':
                            fake_full_path = f'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-a/train/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}/frame'
                        else:
                            fake_full_path = f'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-at/test/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(fake_full_path):
                            file_full_path = os.path.join(fake_full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        if os.path.exists(item[0]) and os.path.exists(fake_full_path):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                            self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    # 捕获所有异常并打印，保证代码不崩溃
                    print(e)
                    num = num+1

            random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method== 'Celeb-DF':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    file_name = item[0].split('/')[-1]
                    underline_count = file_name.count('_')
                    vid_id = item[0].split('/')[-1]
                    if underline_count == 2:
                        vid_id = item[0].split('/')[-1]
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/Celeb_flow_unified/{vid_id}'
                        flow_list=[]
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                flow_list.append(file_full_path)
                        fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF/Celeb-synthesis/{vid_id}/frame'
                        rgb_list = []
                        if os.path.exists(fake_full_path):
                            for file in os.listdir(fake_full_path):
                                file_full_path = os.path.join(fake_full_path, file)
                                if os.path.isfile(file_full_path):
                                    rgb_list.append(file_full_path)
                        else:
                            print(f"{fake_full_path}")
                        if os.path.exists(flow_path) and os.path.exists(fake_full_path):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                            self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                    elif underline_count == 1:
                        vid_id = item[0].split('/')[-1]
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/Celeb_flow_unified/{vid_id}'
                        flow_list=[]
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                flow_list.append(file_full_path)
                        underline_count = vid_id.count('_')
                        real_full_path_celeb = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF/Celeb-real/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(real_full_path_celeb):
                            file_full_path = os.path.join(real_full_path_celeb, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        if os.path.exists(flow_path) and os.path.exists(real_full_path_celeb):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                            self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                    else :
                        vid_id = item[0].split('/')[-1]
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/Celeb_flow_unified/{vid_id}'
                        flow_list=[]
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                flow_list.append(file_full_path)
                        real_full_path_youtube = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF/YouTube-real/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(real_full_path_youtube):
                            file_full_path = os.path.join(real_full_path_youtube, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        
                        if os.path.exists(flow_path) and os.path.exists(real_full_path_youtube):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                            self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1
            if self.split == 'train':
                random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method == 'DFDC':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_id = item[0].split('/')[-1]
                    flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DFDC_flow_unified/{vid_id}'
                    flow_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                    full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/DFDC/test_videos/{vid_id}/frame'
                    rgb_list = []
                    if os.path.exists(full_path):
                        for file in os.listdir(full_path):
                            file_full_path = os.path.join(full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                    else:
                        print(f"{full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1
            if self.split == 'train':
                random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method == 'DFDCP':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_id = item[0].split('/')[-1]
                    label = item[1]  
                    sub_dir = 'real' if label == 0 else 'fake'
                    flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified/{vid_id}'
                    flow_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                    full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/DFDCP/{sub_dir}/{vid_id}'
                    rgb_list = []
                    if os.path.exists(full_path):
                        for file in os.listdir(full_path):
                            file_full_path = os.path.join(full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                    else:
                        print(f"{full_path}")
                    if os.path.exists(item[0]) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1
            if self.split == 'train':
                random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method == 'DFD':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_id = item[0].split('/')[-1]
                    label = item[1]  
                    sub_dir = 'real' if label == 0 else 'fake'
                    flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified/{vid_id}'
                    flow_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                    full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/DFD/{sub_dir}/frame/{vid_id}'
                    rgb_list = []
                    if os.path.exists(full_path):
                        for file in os.listdir(full_path):
                            file_full_path = os.path.join(full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                    else:
                        print(f"{full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1
            if self.split == 'train':
                random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method == 'FFIW':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_id = '/'.join(item[0].split('/')[-3:])
                    vid = item[0].split('/')[-1]
                    label = item[1]  
                    sub_dir = 'real' if label == 0 else 'fake'
                    flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/{vid_id}'
                    flow_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                    full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FFIW/{sub_dir}/frame/{vid}'
                    rgb_list = []
                    if os.path.exists(full_path):
                        for file in os.listdir(full_path):
                            file_full_path = os.path.join(full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                    else:
                        print(f"{full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1
            if self.split == 'train':
                random.shuffle(self.all_list) 
            print('datainfo:',self.split, len(self.all_list), num)
        elif self.method == 'df-40-facedancer':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/facedancer/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-mobileswap':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/mobileswap/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-sadtalker':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/sadtalker/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-uniface':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/uniface/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-simswap':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/simswap/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-inswap':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/inswapper/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        elif self.method == 'df-40-fomm':
            self.all_list=[]
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            for item in tqdm(flow_list):
                try:
                    vid_label = item[1]
                    vid = item[0].split('/')[-1]
                    if vid_label==1:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/DF40_flow_unified/fomm/{vid}'
                        full_path = item[0]
                    else:
                        flow_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid}'
                        full_path = os.path.join(item[0], 'frame')

                    flow_list=[]
                    rgb_list=[]
                    for file in os.listdir(flow_path):
                        file_full_path = os.path.join(flow_path, file)
                        if os.path.isfile(file_full_path):
                            flow_list.append(file_full_path)
                        else:
                            print(f"flow error, {full_path}")
                    for file in os.listdir(full_path):
                        file_full_path = os.path.join(full_path, file)
                        if os.path.isfile(file_full_path):
                            rgb_list.append(file_full_path)
                        else:
                            print(f"rgb error, {file_full_path}")
                    if os.path.exists(flow_path) and os.path.exists(full_path):
                        sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                        sorted_paths_flow = self.sort_image_paths_by_frame_id(flow_list)
                        self.all_list.append([sorted_paths_flow,sorted_paths_rgb,item[1]])
                except Exception as e:       
                    print(e)
                    num = num+1   
            if self.split == 'train':
                random.shuffle(self.all_list) 
        
    def __len__(self):
        return len(self.all_list)    


    def __getitem__(self, index):
        flag = True  #循环标志
        while flag:
            try:
                file_info = self.all_list[index]
                flow_pic_list = file_info[0]
                frame_list_origin   = file_info[1] # 保留原始完整列表
                video_label  = file_info[2]
                video_label = torch.tensor(video_label)

                num_segments = self.num_segments  
                valid_range = len(frame_list_origin) - 4
                if valid_range <= 0:
                    start_indices = [0] * num_segments
                    print('id<0 error')
                    assert 0
                else:
                    start_indices = np.linspace(0, valid_range, num_segments).astype(int)
                segment_rgb_list = []
                segment_flow_list = []
                for start_idx in start_indices:
                    current_frame_list = frame_list_origin[start_idx : start_idx + 4]
                    flow_start = start_idx * 2 - 6 
                    flow_end = start_idx * 2 + 14
                    if flow_start < 0:
                        flow_start = 0
                        flow_end = flow_start + 20
                        flow_segment = flow_pic_list[flow_start:flow_end]
                    elif flow_end > len(flow_pic_list):
                        flow_start = len(flow_pic_list) - 20 
                        if flow_start < 0: flow_start = 0
                        flow_end = len(flow_pic_list)
                        flow_segment = flow_pic_list[flow_start:flow_end]
                    else:
                        flow_segment = flow_pic_list[flow_start:flow_end]
                    flow_tensor_list = []
                    for flow_frame in flow_segment:
                        frame = Image.open(flow_frame)
                        frame = np.asarray(frame)
                        flow_tensor = torch.from_numpy(frame)
                        flow_tensor_list.append(flow_tensor)
                    current_flow_fea = torch.stack(flow_tensor_list) 
                    flow_fea = torch.stack(flow_tensor_list) 
                    all_frames = []
                    additional_targets = {}
                    for i, frame_path in enumerate(current_frame_list):
                        frame = Image.open(frame_path)
                        frame = np.asarray(frame)
                        all_frames.append(frame)
                        if i == 0:
                            tmp_imgs = {"image": all_frames[0]}
                        else:
                            key = f"image{i}"
                            additional_targets[key] = "image"
                            tmp_imgs[key] = all_frames[i]
                    self.transform.add_targets(additional_targets)
                    augmented = self.transform(**tmp_imgs) # 变成字典
                    augmented = OrderedDict(sorted(augmented.items(), key=lambda x: x[0]))
                    augmented_frames = list(augmented.values()) # [Array(H,W,C), ...]
                    augmented_frames = torch.stack(augmented_frames) 
                    processed_rgb = augmented_frames 
                    flow_1 = current_flow_fea[0:10].unsqueeze(0)
                    flow_2 = current_flow_fea[2:12].unsqueeze(0)
                    flow_3 = current_flow_fea[4:14].unsqueeze(0)
                    flow_4 = current_flow_fea[6:16].unsqueeze(0)
                    flow_5 = current_flow_fea[8:18].unsqueeze(0)
                    flow_6 = current_flow_fea[10:20].unsqueeze(0)
                    flow_fea = torch.cat([flow_1, flow_2, flow_3, flow_4, flow_5, flow_6],dim=0)
                    segment_rgb_list.append(processed_rgb)
                    segment_flow_list.append(flow_fea)
                    self.transform.targets = {'image': 'image'} # 假设只保留基础 image
                process_imgs = torch.stack(segment_rgb_list)
                flow_fea = torch.stack(segment_flow_list)
                flag = False
            except Exception as e:
                print('error', e)
                index = random.randint(0, len(self.all_list) - 1)
            # print(flow_fea.shape, process_imgs.shape)
        return {"flow_fea":flow_fea, "process_imgs":process_imgs, "labels":video_label, "index":index}    

