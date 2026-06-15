import os
from pylab import *
import matplotlib.font_manager as fm # to create font
import json
import pandas as pd
import math
from glob import glob
import random
seed_value = 1234 
random.seed(seed_value)
REAL_LABLE = 0
FAKE_LABEL = 1
dataset_root = {
    'FaceForensics_c23_train': 'FF++/face_v1/',
    'FaceForensics': '/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/',
    'Celeb-DF': 'Celeb-DF_frames/face_v1/',
    'DFDC': 'DFDC_frames/face_v1/',
    'FFIW':'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FFIW/',
    'df-40-fomm': 'fomm',
    'df-40-mobileswap': 'mobileswap',
    'df-40-sadtalker': 'sadtalker',
    'df-40-facedancer':'facedancer',
    'df-40-uniface':'uniface',
    'df-40-simswap':'simswap',
    'df-40-inswap':'inswap',
    }


  
def get_data_list(dataset_name, base_root, split, only_real=False):
    # compress = dataset_name.split("_")[1]
    # 修改为：
    parts = dataset_name.split("_")
    compress = parts[1] if len(parts) > 1 else "raw"
    if split == 'train':
        root = os.path.join(base_root, dataset_root[dataset_name+"_train"])
    else:
        root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])  #root=/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/
    dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)#/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++
    #['original_sequences/youtube/raw/videos/953', 0]  
    dataset_info_rgb=[]
    for item in dataset_info:
        if 'original_sequences' in item[0]:
            vid_id = item[0].split('/')[-1]
            real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/original_sequences/youtube/c23/videos/{vid_id}'
            dataset_info_rgb.append([real_full_path, REAL_LABLE])
            # print('11111',dataset_info_rgb[:5])
        else:
            fake_method = item[0].split('/')[1]
            vid_id = item[0].split('/')[-1]
            fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}'
            dataset_info_rgb.append([fake_full_path, FAKE_LABEL])
        
    return dataset_info_rgb,root
# def get_data_list_flow(dataset_name, base_root, split, only_real=False):
#     # compress = dataset_name.split("_")[1]
#     # 修改为：
#     parts = dataset_name.split("_")
#     compress = parts[1] if len(parts) > 1 else "raw"
#     if split == 'train':
#         root = os.path.join(base_root, dataset_root[dataset_name+"_train"])
#     else:
#         root = base_root  #root=/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified
#     dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)#/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++
#     #['original_sequences/youtube/raw/videos/953', 0]  
#     dataset_info_rgb=[]
#     for item in dataset_info:
#         if 'original_sequences' in item[0]:
#             vid_id = item[0].split('/')[-1]
#             real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/original_sequences/youtube/c23/videos/{vid_id}'
#             dataset_info_rgb.append([real_full_path, REAL_LABLE])
#             # print('11111',dataset_info_rgb[:5])
#         else:
#             fake_method = item[0].split('/')[1]
#             vid_id = item[0].split('/')[-1]
#             fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}'
#             dataset_info_rgb.append([fake_full_path, FAKE_LABEL])
        
#     return dataset_info_rgb,root

# def get_data_list_flow(dataset_name, base_root, split, only_real=False):
#     print('====111', dataset_name)
#     # if split == 'train':
#     #     dataset_name = 'FaceForensics_c23'
#     if dataset_name == 'FaceForensics_c23':
#         dataset_info = []
#         compress = 'c23'
#         if split == 'train':
#             root = os.path.join(base_root, dataset_root[dataset_name+"_train"])#/home/guozonghui/data1/project/01-FFD/datasets_processed/FF++/face_v1/
#         else:
#             root = os.path.join(base_root, dataset_root[dataset_name])#/home/guozonghui/data1/project/01-FFD/datasets_processed/FF_frames/face_v1/
#         dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)#root只是为了得到有绝对路径和标签的列表
#         dataset_info_flow = []
#         print(len(dataset_info))  
#         # /ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified
#         for item in dataset_info:
#             if 'original_sequences' in item[0]:
#                 vid_id = item[0].split('/')[-1]
#                 real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/original_sequences/{vid_id}'
#                 dataset_info_flow.append([real_full_path, REAL_LABLE])
#             else:
#                 fake_method = item[0].split('/')[1]
#                 vid_id = item[0].split('/')[-1]
#                 fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/{fake_method}/{vid_id}'
#                 dataset_info_flow.append([fake_full_path, FAKE_LABEL])


#     elif dataset_name == 'Celeb-DF':
#         compress = 'c23'
#         dataset_info_flow = []
#         txt_path = '/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF/List_of_testing_videos.txt'  # 标签文件路径
#         # npy_root = '/data4-16T/liuyingjie/datasets/FFD_flows/test/tsn_processed/'  # npy文件根路径
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#         for line in lines:
#             line = line.strip()  # 去除换行/空格
#             if not line:  # 跳过空行
#                 continue
#             parts = line.split(' ', 1)  # 按第一个空格拆分，避免路径含空格
#             label_str, video_rel_path = parts
#             if len(parts) != 2:
#                 print(f"无效行，跳过：{line}")
#                 continue
#             if label_str =='1':
#                 label = REAL_LABLE
#             else:
#                 label = FAKE_LABEL
            
#             # 提取视频ID（如 "YouTube-real/00170.mp4" → "00170"）
#             video_filename = os.path.basename(video_rel_path)
#             vid_id = os.path.splitext(video_filename)[0]  # 去除后缀：00170
#             full_path = os.path.join('/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF', vid_id)
#             dataset_info_flow.append([full_path, label])


#     elif dataset_name == 'DFDC':
#         compress = 'c23'
#         dataset_info_flow = []
#         video_dir = "/ssd/liuyingjie/datasets/raw_videos/DFDC/test_videos"
#         csv_path = "/ssd/liuyingjie/datasets/ffd-video-data-at/test/DFDC/labels.csv"
#         df = pd.read_csv(csv_path)
#         label_map = dict(zip(df['filename'].str.strip(), df['label'].astype(str).str.strip()))
#         # 3. 遍历npy文件，匹配名称并输出label
#         for video_file in os.listdir(video_dir):
#             if not video_file.endswith('.mp4'):
#                 continue
#             # 提取纯视频名（去掉.npy后缀）
#             video_name = video_file
#             label = label_map.get(video_name, '未知')
#             if label =='1':
#                 label = FAKE_LABEL
#             else:
#                 label = REAL_LABLE
#             video_full_path = os.path.join(video_dir, video_file).replace('.mp4', '')
#             dataset_info_flow.append([video_full_path, label])

        
#     elif dataset_name == 'DFDCP':
#         compress = 'c23'
#         dataset_info_flow = []
#         root = "/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified"
#         csv_path = "/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified/labels.csv"
#         df = pd.read_csv(csv_path)
#         label_map = dict(zip(
#         df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
#         df['label'].astype(str).str.strip()
#         ))
#         for dir in os.listdir(root):
#             video_name = dir
#             label = label_map.get(video_name, '未知')
#             # print(f"视频名: {video_name} → Label: {label}")
#             if label =='1':
#                 label = FAKE_LABEL
#             else:
#                 label = REAL_LABLE
#             full_path = f"/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified/{video_name}"
#             if os.path.exists(full_path):
#                 dataset_info_flow.append([full_path, label])
#             else:
#                 print(f"警告,跳过：{full_path}")
#         print(f"\n共匹配到 {len(dataset_info_flow)} 个视频")   

#     elif dataset_name == 'DFD':
#         compress = 'c23'
#         dataset_info_flow = []
#         root = "/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified"
#         csv_path = "/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified/labels.csv"
#         df = pd.read_csv(csv_path)
#         label_map = dict(zip(
#         df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
#         df['label'].astype(str).str.strip()
#         ))
#         for dir in os.listdir(root):
#             video_name = dir
#             label = label_map.get(video_name, '未知')
#             # print(f"视频名: {video_name} → Label: {label}")
#             if label =='1':
#                 label = FAKE_LABEL
#             else:
#                 label = REAL_LABLE
#             full_path = f"/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified/{video_name}"
#             if os.path.exists(full_path):
#                 dataset_info_flow.append([full_path, label])
#             else:
#                 print(f"警告,跳过：{full_path}")
#         print(f"\n共匹配到 {len(dataset_info_flow)} 个视频")   
#     # elif dataset_name == 'df-40-facedancer':
#     #     compress = 'c23'
#     #     dataset_info_flow = []
#     #     split = 'test'
#     #     real_root = "/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences"
#     #     real_path = glob(real_root + '/*', recursive=True) #字符串路径变成文件路径列表
#     #     for i,path in enumerate(real_path):
#     #         dataset_info_flow.append((path, REAL_LABLE))
#     #     # dataset_info_flow = get_FF_list(real_root, split, compress='c23', only_real=True)
#     #     fake_root = os.path.join(base_root, dataset_root[dataset_name])
#     #     fake_path = glob(fake_root + '/*', recursive=True) 
#     #     for i,path in enumerate(fake_path):
#     #         dataset_info_flow.append((path, FAKE_LABEL))
#     elif "df-40" in dataset_name:
#         split = 'test'
#         root = os.path.join(base_root, dataset_root['FaceForensics'])
#         dataset_info = get_FF_list_test(root, split, compress='c23', only_real=True)
#         fake_root = '/ssd/liuyingjie/datasets/ffd-video-data-at/test/DF40'
#         fake_root = os.path.join(fake_root, dataset_root[dataset_name])

#         fake_path = glob(fake_root + '/frame/*', recursive=True)

#         for i,path in enumerate(fake_path):
#             dataset_info.append((path.replace(base_root,'').replace('/FF++/',''), FAKE_LABEL))

#         return dataset_info

#     elif dataset_name == 'FFIW':
#         assert 0
#         compress = 'c23'
#         dataset_info_flow = []
#         root = "/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified"
#         csv_path = "/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/labels.csv"
#         df = pd.read_csv(csv_path)
#         label_map = dict(zip(
#     df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
#     df['label'].astype(str).str.strip()
#     ))
#         fake_dirs = os.path.join(root, "fake", "flow")
#         for fake_dir in os.listdir(fake_dirs):
#             video_name = os.path.splitext(fake_dir)[0]
#             label = FAKE_LABEL
#             full_path = f"/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/fake/flow/{video_name}"
#             if os.path.exists(full_path):
#                 dataset_info_flow.append([full_path, label])
#             else:
#                 print(f"警告,跳过：{full_path}")

#         real_dirs = os.path.join(root, "real", "flow")
#         for real_dir in os.listdir(real_dirs):
#             video_name = os.path.splitext(real_dir)[0]
#             label = REAL_LABLE
#             full_path = f"/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/real/flow/{video_name}"
#             if os.path.exists(full_path):
#                 dataset_info_flow.append([full_path, label])
#             else:
#                 print(f"警告,跳过：{full_path}")

#     else:
#         raise ValueError(f"不支持的数据集名称：{dataset_name}")
#     return dataset_info_flow  #重点返回的是dataset_info_flow，是一个存放所有视频npy路径以及它们对应的标签
        # for line in text:
        #     path,label='',''line
        #     dataset_info_flow.append([fake_full_path, FAKE_LABEL])
        # print(len(dataset_info_flow))
    # else:
    #     print('not support')
    # return dataset_info_flow, base_root
def get_data_list_flow_1(dataset_name, base_root, split, only_real=False):
    if split == 'train':
        dataset_name = 'FaceForensics_c23'
    if dataset_name == 'FaceForensics_c23':
        dataset_info = []
        compress = 'c23'
        if split == 'train':
            root = os.path.join(base_root, dataset_root[dataset_name+"_train"])#/home/guozonghui/data1/project/01-FFD/datasets_processed/FF++/face_v1/
        else:
            root = base_root#'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified'
        dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)#root只是为了得到有绝对路径和标签的列表
        dataset_info_flow = []
        print(len(dataset_info))  
        # /ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified
        for item in dataset_info:
            if 'original_sequences' in item[0]:
                vid_id = item[0].split('/')[-1]
                real_full_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences/{vid_id}'
                dataset_info_flow.append([real_full_path, REAL_LABLE])
            else:
                fake_method = item[0].split('/')[1]
                vid_id = item[0].split('/')[-1]
                fake_full_path = f'/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/{fake_method}/{vid_id}'
                dataset_info_flow.append([fake_full_path, FAKE_LABEL])
    else:
        raise ValueError(f"不支持的数据集名称：{dataset_name}")
    return dataset_info_flow  #重点返回的是dataset_info_flow，是一个存放所有视频npy路径以及它们对应的标签
        # for line in text:
        #     path,label='',''line
        #     dataset_info_flow.append([fake_full_path, FAKE_LABEL])
        # print(len(dataset_info_flow))
    # else:
    #     print('not support')
    # return dataset_info_flow, base_root
 
# def get_data_list(dataset_name, base_root, split, only_real=False):
#      # 定义需要遍历的两个核心目录（只从这两个目录取子文件夹）
#     core_dirs = [
#         # 伪造视频核心目录（只遍历该目录下的子文件夹）
#         '/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-a/train/FF++/manipulated_sequences/Deepfakes/c23/videos',
#         # 真实视频核心目录（只遍历该目录下的子文件夹）
#         '/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-a/train/FF++/original_sequences'
#     ]
#     dataset_info = []
#     if "FaceForensics" in dataset_name:
#         full_path = f'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-a/train/FF++'
#         # 遍历目录下的所有视频子目录（每个子目录对应一个视频）
#     # for video_dir in os.listdir(full_path):
#     #     # 拼接完整的视频路径
#     #     video_path = os.path.join(full_path, video_dir)
#     #     print('111111',video_path)
#     #     # 过滤：只保留目录（排除文件）
#     #     if not os.path.isdir(video_path):
#     #         continue
#         # 遍历FF++下的子目录（如manipulated_sequences/original_sequences）
#     for seq_type in os.listdir(full_path):
#         seq_type_path = os.path.join(full_path, seq_type)
#         if not os.path.isdir(seq_type_path):
#             continue
        
#         # 遍历伪造方法目录（如Deepfakes）
#         for method in os.listdir(seq_type_path):
#             method_path = os.path.join(seq_type_path, method)
#             if not os.path.isdir(method_path):
#                 continue
            
#             # 遍历压缩级别目录（如c23）
#             for compress in os.listdir(method_path):
#                 compress_path = os.path.join(method_path, compress)
#                 if not os.path.isdir(compress_path):
#                     continue
                
#                 # 遍历videos目录（存放视频子目录）
#                 videos_path = os.path.join(compress_path, 'videos')
#                 if not os.path.exists(videos_path):
#                     continue  # 跳过无videos子目录的路径
                
#                 # 遍历每个视频子目录（如001_870）
#                 for video_dir in os.listdir(videos_path):
#                     video_path = os.path.join(videos_path, video_dir)
#                     if not os.path.isdir(video_path):
#                         continue
                    
#                     # 拼接frame子目录路径（核心：获取每个视频的frame目录）
#                     frame_dir = os.path.join(video_path, 'frame')
#                     # print('66666666666',frame_dir)
#                     if not os.path.exists(frame_dir):
#                         print(f"警告：无frame目录，跳过：{video_path}")
#                         continue
#                     dataset_info.append(frame_dir)
#     return dataset_info          
#         # compress = dataset_name.split("_")[1]
#         # if split == 'train':
#         #     root = os.path.join(base_root, dataset_root[dataset_name+"_train"])
#         # else:
#         #     root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])
#         # dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)
#         # 将 (视频路径, 标签) 加入列表
#         # dataset_info.append(video_path)
    
#     # elif dataset_name == 'Celeb-DF' and split=='test':
#     #     root = os.path.join(base_root, dataset_root[dataset_name])
#     #     video_list_txt = os.path.join(root, 'List_of_testing_videos.txt')
#     #     with open(video_list_txt) as f:
#     #         for data in f:
#     #             line=data.split()
#     #             dataset_info.append((line[1][:-4],FAKE_LABEL-int(line[0])))
#     # elif dataset_name == 'DFDC' and split=='test':
#     #     root = os.path.join(base_root, dataset_root[dataset_name])
#     #     label=pd.read_csv(root+'labels.csv',delimiter=',')
#     #     dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
#     #     root = root+'test_videos/'



def get_data_list_flow(dataset_name, base_root, split, only_real=False):
    # if split == 'train':
    #     dataset_name = 'FaceForensics_c23'
    if dataset_name == 'FaceForensics_c23':
        dataset_info = []
        compress = 'c23'
        if split == 'train':
            root = os.path.join(base_root, dataset_root[dataset_name+"_train"])#/home/guozonghui/data1/project/01-FFD/datasets_processed/FF++/face_v1/
        else:
            root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])#/home/guozonghui/data1/project/01-FFD/datasets_processed/FF_frames/face_v1/
        dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)#root只是为了得到有绝对路径和标签的列表
        dataset_info_flow = []
        print(len(dataset_info))
        for item in dataset_info:
            if 'original_sequences' in item[0]:
                vid_id = item[0].split('/')[-1]
                real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/original_sequences/{vid_id}'
                dataset_info_flow.append([real_full_path, REAL_LABLE])
            else:
                fake_method = item[0].split('/')[1]
                vid_id = item[0].split('/')[-1]
                fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/{fake_method}/{vid_id}'
                dataset_info_flow.append([fake_full_path, FAKE_LABEL])


    elif dataset_name == 'Celeb-DF':
        compress = 'c23'
        dataset_info_flow = []
        txt_path = '/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF/List_of_testing_videos.txt'  # 标签文件路径
        # npy_root = '/data4-16T/liuyingjie/datasets/FFD_flows/test/tsn_processed/'  # npy文件根路径
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()  # 去除换行/空格
            if not line:  # 跳过空行
                continue
            parts = line.split(' ', 1)  # 按第一个空格拆分，避免路径含空格
            label_str, video_rel_path = parts
            if len(parts) != 2:
                print(f"无效行，跳过：{line}")
                continue
            if label_str =='1':
                label = REAL_LABLE
            else:
                label = FAKE_LABEL
            
            # 提取视频ID（如 "YouTube-real/00170.mp4" → "00170"）
            video_filename = os.path.basename(video_rel_path)
            vid_id = os.path.splitext(video_filename)[0]  # 去除后缀：00170
            full_path = os.path.join('/ssd/liuyingjie/datasets/ffd-video-data-at/test/Celeb-DF', vid_id)
            dataset_info_flow.append([full_path, label])


    elif dataset_name == 'DFDC':
        compress = 'c23'
        dataset_info_flow = []
        video_dir = "/ssd/liuyingjie/datasets/raw_videos/DFDC/test_videos"
        csv_path = "/ssd/liuyingjie/datasets/ffd-video-data-at/test/DFDC/labels.csv"
        df = pd.read_csv(csv_path)
        label_map = dict(zip(df['filename'].str.strip(), df['label'].astype(str).str.strip()))
        # 3. 遍历npy文件，匹配名称并输出label
        for video_file in os.listdir(video_dir):
            if not video_file.endswith('.mp4'):
                continue
            # 提取纯视频名（去掉.npy后缀）
            video_name = video_file
            label = label_map.get(video_name, '未知')
            if label =='1':
                label = FAKE_LABEL
            else:
                label = REAL_LABLE
            video_full_path = os.path.join(video_dir, video_file).replace('.mp4', '')
            dataset_info_flow.append([video_full_path, label])

        
    elif dataset_name == 'DFDCP':
        compress = 'c23'
        dataset_info_flow = []
        root = "/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified"
        csv_path = "/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified/labels.csv"
        df = pd.read_csv(csv_path)
        label_map = dict(zip(
    df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
    df['label'].astype(str).str.strip()
    ))
        for dir in os.listdir(root):
            video_name = dir
            label = label_map.get(video_name, '未知')
            # print(f"视频名: {video_name} → Label: {label}")
            if label =='1':
                label = FAKE_LABEL
            else:
                label = REAL_LABLE
            full_path = f"/ssd/liuyingjie/datasets/ffd_flow/DFDCP_flow_unified/{video_name}"
            if os.path.exists(full_path):
                dataset_info_flow.append([full_path, label])
            else:
                print(f"警告,跳过：{full_path}")
        # print(f"\n共匹配到 {len(dataset_info_flow)} 个视频")   

    elif dataset_name == 'DFD':
        compress = 'c23'
        dataset_info_flow = []
        root = "/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified"
        csv_path = "/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified/labels.csv"
        df = pd.read_csv(csv_path)
        label_map = dict(zip(
    df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
    df['label'].astype(str).str.strip()
    ))
        for dir in os.listdir(root):
            video_name = dir
            label = label_map.get(video_name, '未知')
            # print(f"视频名: {video_name} → Label: {label}")
            if label =='1':
                label = FAKE_LABEL
            else:
                label = REAL_LABLE
            full_path = f"/ssd/liuyingjie/datasets/ffd_flow/DFD_flow_unified/{video_name}"
            if os.path.exists(full_path):
                dataset_info_flow.append([full_path, label])
            else:
                print(f"警告,跳过：{full_path}")
        # print(f"\n共匹配到 {len(dataset_info_flow)} 个视频")   
    # elif dataset_name == 'df-40-facedancer':
    #     compress = 'c23'
    #     dataset_info_flow = []
    #     split = 'test'
    #     real_root = "/ssd/liuyingjie/datasets/ffd_flow/FF++_flow_unified/original_sequences"
    #     real_path = glob(real_root + '/*', recursive=True) #字符串路径变成文件路径列表
    #     for i,path in enumerate(real_path):
    #         dataset_info_flow.append((path, REAL_LABLE))
    #     # dataset_info_flow = get_FF_list(real_root, split, compress='c23', only_real=True)
    #     fake_root = os.path.join(base_root, dataset_root[dataset_name])
    #     fake_path = glob(fake_root + '/*', recursive=True) 
    #     for i,path in enumerate(fake_path):
    #         dataset_info_flow.append((path, FAKE_LABEL))
    elif "df-40" in dataset_name:
        split = 'test'
        root = os.path.join(base_root, dataset_root['FaceForensics'])
        dataset_info = get_FF_list_test(root, split, compress='c23', only_real=True)
        fake_root = '/ssd/liuyingjie/datasets/ffd-video-data-at/test/DF40'
        fake_root = os.path.join(fake_root, dataset_root[dataset_name])

        fake_path = glob(fake_root + '/frame/*', recursive=True)

        for i,path in enumerate(fake_path):
            dataset_info.append((path.replace(base_root,'').replace('/FF++/',''), FAKE_LABEL))

        return dataset_info

    elif dataset_name == 'FFIW':
        compress = 'c23'
        dataset_info_flow = []
        root = "/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified"
        csv_path = "/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/labels.csv"
        df = pd.read_csv(csv_path)
        label_map = dict(zip(
    df['video'].astype(str).str.strip(),  # 强制转字符串，再去空格
    df['label'].astype(str).str.strip()
    ))
        fake_dirs = os.path.join(root, "fake", "flow")
        for fake_dir in os.listdir(fake_dirs):
            video_name = os.path.splitext(fake_dir)[0]
            label = FAKE_LABEL
            full_path = f"/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/fake/flow/{video_name}"
            if os.path.exists(full_path):
                dataset_info_flow.append([full_path, label])
            else:
                print(f"警告,跳过：{full_path}")

        real_dirs = os.path.join(root, "real", "flow")
        for real_dir in os.listdir(real_dirs):
            video_name = os.path.splitext(real_dir)[0]
            label = REAL_LABLE
            full_path = f"/ssd/liuyingjie/datasets/ffd_flow/FFIW_flow_unified/real/flow/{video_name}"
            if os.path.exists(full_path):
                dataset_info_flow.append([full_path, label])
            else:
                print(f"警告,跳过：{full_path}")

    else:
        raise ValueError(f"不支持的数据集名称：{dataset_name}")
    return dataset_info_flow  #重点返回的是dataset_info_flow，是一个存放所有视频npy路径以及它们对应的标签
        # for line in text:
        #     path,label='',''line
        #     dataset_info_flow.append([fake_full_path, FAKE_LABEL])
        # print(len(dataset_info_flow))
    # else:
    #     print('not support')
    # return dataset_info_flow, base_root
  


def get_FF_list(root, split, compress='c23', only_real=False):#针对FF++，生成「视频文件路径 + 真假标签」的列表
    root = '/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++'
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join(root,'original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), REAL_LABLE] for x in real_names]
    else:#用这个，真实伪造视频都用到
        real_names = []
        fake_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])#每一对真实视频会生成双向的伪造视频，item是二维列表["000", "001"]、["002", "003"]
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in ff_fake_types:
            fake_video_dir = os.path.join('manipulated_sequences', method, compress, 'videos')
            for x in fake_names:
                dataset_info.append((os.path.join(fake_video_dir,x),FAKE_LABEL))
    return dataset_info

def get_FF_list_test(root, split, compress='c23', only_real=False):#针对FF++，生成「视频文件路径 + 真假标签」的列表
    root = '/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++'
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join(root,'original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), REAL_LABLE] for x in real_names]
    else:#用这个，真实伪造视频都用到
        real_names = []
        fake_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])#每一对真实视频会生成双向的伪造视频，item是二维列表["000", "001"]、["002", "003"]
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in ff_fake_types:
            fake_video_dir = os.path.join('manipulated_sequences', method, compress, 'videos')
            for x in fake_names:
                dataset_info.append((os.path.join(fake_video_dir,x),FAKE_LABEL))
    return dataset_info


def check_frame_len(video_len, num_segments):
    inner_index = list(range(video_len))
    pad_length = math.ceil((num_segments-video_len)/2)
    post_module = inner_index[1:-1][::-1] + inner_index
    l_post = len(post_module)
    post_module = post_module * (pad_length // l_post + 1)
    post_module = post_module[:pad_length]
    assert len(post_module) == pad_length
    pre_module = inner_index + inner_index[1:-1][::-1]
    l_pre = len(post_module)
    pre_module = pre_module * (pad_length // l_pre + 1)
    pre_module = pre_module[-pad_length:]
    assert len(pre_module) == pad_length
    sampled_clip_idxs = pre_module + inner_index + post_module
    sampled_clip_idxs = sampled_clip_idxs[:num_segments]
    return sampled_clip_idxs



    """
    随机策略光流平滑器
    输入: [N, H, W] Tensor (单通道光流/深度/差异图)
    功能: 转为 NumPy -> 随机选择一种策略 -> 对 N 帧应用完全一致的参数 -> 转回 Tensor
    """
    
    def __init__(self, p_smooth=0.8):
        self.p_smooth = p_smooth
        
        self.strategy_names = [
            'bilateral',
            'edge_aware',
            'frequency',
            'local_consistency',
            'light_gaussian',
            'strong_gaussian',
            'anisotropic',
            'tv_denoising'
        ]
        
        self.strategy_funcs = {
            'bilateral': self._bilateral_smooth,
            'edge_aware': self._edge_aware_smooth,
            'frequency': self._frequency_smooth,
            'local_consistency': self._local_consistency_smooth,
            'light_gaussian': self._light_gaussian_smooth,
            'strong_gaussian': self._strong_gaussian_smooth,
            'anisotropic': self._anisotropic_diffusion_smooth,
            'tv_denoising': self._tv_denoising_smooth,
        }
        
        # 权重均匀
        self.strategy_weights = [1.0] * len(self.strategy_names)

    def smooth(self, flow_input, is_fake=True):
        """
        Args:
            flow_input: [N, H, W] (Batch) 或 [H, W] (Single) 的 torch.Tensor
            is_fake: bool, 是否应用平滑
        Returns:
            torch.Tensor, 形状和设备与输入一致
        """
        # --- 1. Tensor 预处理 ---
        is_tensor = False
        device = None
        dtype = None
        input_dim = 0
        
        if torch.is_tensor(flow_input):
            is_tensor = True
            device = flow_input.device
            dtype = flow_input.dtype
            input_dim = flow_input.dim()
            
            # 转为 Numpy (CPU)
            flows_np = flow_input.detach().cpu().numpy()
        else:
            # 兼容直接传 numpy 的情况
            flows_np = flow_input
            input_dim = flows_np.ndim

        # 规范化为 List 形式以便循环处理，同时标记是否需要还原 Batch 维度
        input_type = 'single'
        if input_dim == 3: # [N, H, W]
            input_type = 'batch'
            flow_list = list(flows_np)
        elif input_dim == 2: # [H, W]
            input_type = 'single'
            flow_list = [flows_np]
        else:
            raise ValueError(f"Input shape must be [N, H, W] or [H, W], got {flow_input.shape}")

        # --- 2. 策略决策 (Global Decision) ---
        # 针对整个 Batch 做一次决定，保证 N 帧的一致性
        do_smooth = False
        strategy_name = None
        params = {}
        alpha = 1.0

        if not is_fake:
            # 真实样本：小概率轻微平滑
            if random.random() < 0.3:
                do_smooth = True
                strategy_name = 'light_gaussian'
                params = self._get_strategy_params(strategy_name)
        else:
            # 伪造样本
            if random.random() <= self.p_smooth:
                do_smooth = True
                strategy_name = random.choices(self.strategy_names, weights=self.strategy_weights)[0]
                params = self._get_strategy_params(strategy_name)
                alpha = random.uniform(0.6, 0.9)

        # --- 3. 执行平滑 (Numpy 处理) ---
        if not do_smooth:
            # 不平滑直接返回原数据
            result_np = flows_np
        else:
            processed_flows = []
            smooth_func = self.strategy_funcs[strategy_name]

            for flow in flow_list:
                # 确保是 float32 且内存连续 (cv2 某些函数对 stride 有要求)
                flow_float = np.ascontiguousarray(flow.astype(np.float32))
                
                # 应用固定的 params
                smoothed = smooth_func(flow_float, **params)
                
                # 混合原图
                if alpha < 1.0:
                    smoothed = alpha * smoothed + (1 - alpha) * flow_float
                
                processed_flows.append(smoothed)
            
            # 还原为数组结构
            if input_type == 'batch':
                result_np = np.array(processed_flows)
            else:
                result_np = processed_flows[0]

        # --- 4. 还原为 Tensor ---
        if is_tensor:
            #转回 tensor -> 还原到原设备 -> 还原数据类型
            result_tensor = torch.from_numpy(result_np).to(device).to(dtype)
            return result_tensor
        else:
            return result_np

    # ---------- 参数生成器 (复用之前逻辑) ----------
    def _get_strategy_params(self, name):
        params = {}
        if name == 'bilateral':
            params['d'] = random.choice([3, 5, 7])
            params['sigma_color'] = random.uniform(0.5, 1.5)
            params['sigma_space'] = random.uniform(3, 7)
        elif name == 'edge_aware':
            params['edge_percentile'] = 85
            params['k_flat'] = random.choice([5, 7])
            params['k_edge'] = random.choice([3, 5])
        elif name == 'frequency':
            params['cutoff'] = random.randint(20, 60)
            params['use_gaussian'] = random.random() > 0.5
        elif name == 'local_consistency':
            params['kernel_size'] = random.choice([3, 5])
        elif name == 'light_gaussian':
            params['kernel_size'] = random.choice([3, 5])
            params['sigma'] = random.uniform(0.5, 1.0)
        elif name == 'strong_gaussian':
            params['kernel_size'] = random.choice([7, 9, 11])
            params['sigma'] = random.uniform(1.5, 3.0)
            params['add_noise'] = random.random() > 0.5
            if params['add_noise']:
                params['noise_level'] = random.uniform(0.01, 0.05)
        elif name == 'anisotropic':
            params['iterations'] = 3
            params['k'] = 0.1
        elif name == 'tv_denoising':
            params['lambda_tv'] = 0.1
            params['iterations'] = 10
        return params

    # ---------- 具体平滑策略 (单通道 Numpy 实现) ----------
    
    def _bilateral_smooth(self, flow, d, sigma_color, sigma_space):
        return cv2.bilateralFilter(flow, d, sigma_color, sigma_space)
    
    def _edge_aware_smooth(self, flow, edge_percentile, k_flat, k_edge):
        gx, gy = np.gradient(flow)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_threshold = np.percentile(grad_mag, edge_percentile)
        edge_mask = grad_mag > edge_threshold
        
        flow_smooth = flow.copy()
        if np.sum(~edge_mask) > 0:
            flat_blur = cv2.GaussianBlur(flow, (k_flat, k_flat), 1.5)
            flow_smooth[~edge_mask] = flat_blur[~edge_mask]
        
        edge_blur = cv2.GaussianBlur(flow, (k_edge, k_edge), 0.5)
        flow_smooth[edge_mask] = edge_blur[edge_mask]
        return flow_smooth
    
    def _frequency_smooth(self, flow, cutoff, use_gaussian):
        fft = np.fft.fft2(flow)
        fft_shifted = np.fft.fftshift(fft)
        rows, cols = flow.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.float32)
        y, x = np.ogrid[:rows, :cols]
        if use_gaussian:
            mask = 1 - np.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * (cutoff/2)**2))
        else:
            mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
            mask[mask_area] = 0
        return np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted * mask)))
    
    def _local_consistency_smooth(self, flow, kernel_size):
        gx, gy = np.gradient(flow)
        grad_mag = np.sqrt(gx**2 + gy**2)
        anomaly_mask = grad_mag > np.percentile(grad_mag, 90)
        flow_smooth = flow.copy()
        if np.sum(anomaly_mask) > 0:
            median = cv2.medianBlur(flow, kernel_size)
            flow_smooth[anomaly_mask] = median[anomaly_mask]
        return flow_smooth
    
    def _light_gaussian_smooth(self, flow, kernel_size, sigma):
        return cv2.GaussianBlur(flow, (kernel_size, kernel_size), sigma)
    
    def _strong_gaussian_smooth(self, flow, kernel_size, sigma, add_noise=False, noise_level=0.0):
        smoothed = cv2.GaussianBlur(flow, (kernel_size, kernel_size), sigma)
        if add_noise:
            noise = np.random.randn(*smoothed.shape) * noise_level * np.std(smoothed)
            smoothed += noise
        return smoothed
    
    def _anisotropic_diffusion_smooth(self, flow, iterations, k):
        img = flow.copy()
        for _ in range(iterations):
            ix = np.roll(img, -1, axis=1) - img
            iy = np.roll(img, -1, axis=0) - img
            grad = np.sqrt(ix**2 + iy**2)
            c = 1.0 / (1.0 + (grad / k)**2)
            img = img + 0.25 * (c * ix + c * iy)
        return img
    
    def _tv_denoising_smooth(self, flow, lambda_tv, iterations):
        img = flow.copy()
        for _ in range(iterations):
            ix = np.roll(img, -1, axis=1) - img
            iy = np.roll(img, -1, axis=0) - img
            grad_mag = np.sqrt(ix**2 + iy**2 + 1e-8)
            img = img - lambda_tv * (ix / grad_mag + iy / grad_mag)
        return img