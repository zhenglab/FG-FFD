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
exist_bi = True
import datasets.utils.blend as B
from datasets.utils.initialize import *
from datasets.utils.funcs import IoUfrom2bboxes,crop_face,RandomDownScale
from datasets.utils.mask_generator import MaskingGenerator
from datasets.lib.bi_online_generation import random_get_hull, random_get_hull_custom
import datasets.augmentations.masking as Masking
import logging
import albumentations as alb
import cv2


class Image_dataset_random_rgbwflow_s1_tsn_SBI_s2(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_segments=8,
                 num_segments_flow=8,
                 transform=None,
                 cutout=False,
                 is_sbi=False,
                 image_size=224,
                 method=None):
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
        self.source_transforms = self.get_source_transforms()

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
        if self.method == 'FaceForensics_c23':
            self.all_list = []
            flow_list = dataloader_util.get_data_list_flow(self.method, self.root, self.split, only_real=False)
            num = 0
            cache_dir = "./path_cache"
            os.makedirs(cache_dir, exist_ok=True)
            if not hasattr(self, 'frame_pattern'):
                import re
                self.frame_pattern = re.compile(r'(\d+)')
            for item in tqdm(flow_list, desc="Processing dataset with cache"):
                try:
                    vid_id = item[0].split('/')[-1]
                    if 'original_sequences' in item[0]:
                        cache_suffix = "original"
                    else:
                        fake_method = item[0].split('/')[-2]
                        cache_suffix = f"fake_{fake_method}"
                    cache_path = os.path.join(cache_dir, f"{self.method}_{self.split}_{vid_id}_{cache_suffix}.pkl")
                    if os.path.exists(cache_path):
                        with open(cache_path, 'rb') as f:
                            sorted_paths_flow, sorted_paths_rgb, label = pickle.load(f)
                        self.all_list.append([sorted_paths_flow, sorted_paths_rgb, label])
                        continue
                    if 'original_sequences' in item[0]:
                        flow_path = f'/ssd/liuyingjie/data/ffd_flow/original_sequences/{vid_id}'
                        video_flow_list = []
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                video_flow_list.append(file_full_path)
                        if self.split == 'train':
                            real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/original_sequences/youtube/c23/videos/{vid_id}/frame'
                        else:
                            real_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/original_sequences/youtube/c23/videos/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(real_full_path):
                            file_full_path = os.path.join(real_full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        if os.path.exists(flow_path) and os.path.exists(real_full_path):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(video_flow_list)
                            label = item[1]
                            self.all_list.append([sorted_paths_flow, sorted_paths_rgb, label])
                    else:
                        fake_method = item[0].split('/')[-2]
                        flow_path = f'/ssd/liuyingjie/data/ffd_flow/{fake_method}/{vid_id}'
                        video_flow_list = []
                        for file in os.listdir(flow_path):
                            file_full_path = os.path.join(flow_path, file)
                            if os.path.isfile(file_full_path):
                                video_flow_list.append(file_full_path)
                        if self.split == 'train':
                            fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-a/train/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}/frame'
                        else:
                            fake_full_path = f'/ssd/liuyingjie/datasets/ffd-video-data-at/test/FF++/manipulated_sequences/{fake_method}/c23/videos/{vid_id}/frame'
                        rgb_list = []
                        for file in os.listdir(fake_full_path):
                            file_full_path = os.path.join(fake_full_path, file)
                            if os.path.isfile(file_full_path):
                                rgb_list.append(file_full_path)
                        if os.path.exists(flow_path) and os.path.exists(fake_full_path):
                            sorted_paths_rgb = self.sort_image_paths_by_frame_id(rgb_list)
                            sorted_paths_flow = self.sort_image_paths_by_frame_id(video_flow_list)
                            label = item[1]
                            self.all_list.append([sorted_paths_flow, sorted_paths_rgb, label])
                    if 'sorted_paths_flow' in locals() and 'sorted_paths_rgb' in locals():
                        with open(cache_path, 'wb') as f:
                            pickle.dump((sorted_paths_flow, sorted_paths_rgb, label), f)
                except Exception as e:       
                    print(f"Error processing vid_id {vid_id}: {e}")
                    num += 1
            print(f"Total error videos: {num}")
            print(f"Dataset parsing completed, total valid videos: {len(self.all_list)}")


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
                    start_indices = np.random.randint(0, valid_range, size=num_segments)
                    start_indices = np.sort(start_indices)
                segment_rgb_list = []
                segment_flow_list = []
                for start_idx in start_indices:
                    current_frame_list = frame_list_origin[start_idx : start_idx + 4]
                    flow_start = start_idx * 2 - 4 
                    flow_end = start_idx * 2 + 16
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

                    # if video_label ==1:
                    #     for i, filename_frame in enumerate(current_frame_list):
                    #         frame = Image.open(filename_frame)
                    #         frame = np.asarray(frame)
                    #         landmark_name = filename_frame.replace('frame', 'landmarks').replace('png', 'npy')
                    #         frame_aug, mask = self.augment_frame(frame, 0, 'SBI', video_label, filename_frame, landmark_name, vid_mask=None)
                    #         all_frames.append(frame_aug)
                    #         if i == 0:
                    #             tmp_imgs = {"image": all_frames[0]}
                    #         else:
                    #             key = f"image{i}"
                    #             additional_targets[key] = "image"
                    #             tmp_imgs[key] = all_frames[i]
                    # else:
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

                    # for i, frame_path in enumerate(current_frame_list):
                    #     frame = Image.open(frame_path)
                    #     frame = np.asarray(frame)
                    #     all_frames.append(frame)
                    #     if i == 0:
                    #         tmp_imgs = {"image": all_frames[0]}
                    #     else:
                    #         key = f"image{i}"
                    #         additional_targets[key] = "image"
                    #         tmp_imgs[key] = all_frames[i]


                    self.transform.add_targets(additional_targets)
                    augmented = self.transform(**tmp_imgs) # 变成字典
                    augmented = OrderedDict(sorted(augmented.items(), key=lambda x: x[0]))
                    augmented_frames = list(augmented.values()) # [Array(H,W,C), ...]
                    augmented_frames = torch.stack(augmented_frames) 
                    processed_rgb = augmented_frames
                    processed_rgb = self.cutout(processed_rgb)


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
                # print('error', e)
                index = random.randint(0, len(self.all_list) - 1)
            # print(flow_fea.shape, process_imgs.shape)
        return {"flow_fea":flow_fea, "process_imgs":process_imgs, "labels":video_label, "index":index}    



    def __len__(self):
        return len(self.all_list)
    def augment_frame(self, frame,is_aug, blending_type, video_labels,filename_frame, landmark_name, vid_mask=None):
        if vid_mask is None:
            if blending_type == 'SBI':
                if is_aug == 0:
                    _,frame,mask=self.self_blending_custom(filename_frame,landmark_name,frame, method='SBI')
                else:
                    mask = torch.zeros((224,224,1))
                    frame = self.source_transforms(image=frame.astype(np.uint8))['image']
            elif blending_type == 'components':
                if is_aug == 0:
                    _,frame,mask=self.self_blending_custom(filename_frame,landmark_name,frame, method='components')
                else:
                    mask = torch.zeros((224,224,1))
                    frame = self.source_transforms(image=frame.astype(np.uint8))['image']
            elif blending_type == 'sladd':
                if is_aug == 0:
                    landmark = np.load(landmark_name)[0]
                    mask = self.masking_sladd.compute_mask(frame, landmarks=landmark, idx=0)
                    mask = mask.astype(np.float32)
                    frame_ori,frame,mask=self.self_blending_by_mask(frame,mask)
                else:
                    mask = torch.zeros((224,224,1))
                    frame = self.source_transforms(image=frame.astype(np.uint8))['image']
        else:
            _,frame,mask=self.self_blending_by_mask(frame,vid_mask)
        return frame, mask

    def video_masks_generated(self, is_aug,file_info):
        filename_frame = file_info[0][0]
        video_labels = file_info[0][1]
        landmark_name = file_info[0][2]
        frame = Image.open(filename_frame)
        frame = np.asarray(frame)

        if self.blending_type == 'SBI':
            '''self_blending_SBI'''
            # if video_labels == 0 and is_aug == 0:
            if is_aug == 0:
                landmark = np.load(landmark_name)[0]
                mask = self.self_blending_mask(frame, landmark)
            else:
                mask = torch.zeros((224,224,1))
        else:
            '''self_blending_custom_masking'''
            # if video_labels == 0 and is_aug == 0:
            if is_aug == 0:
                landmark = np.load(landmark_name)[0]
                mask = self.masking.compute_mask(frame, landmarks=landmark, idx=self.rand_c)
                mask = mask/255
            else:
                mask = torch.zeros((224,224,1))
        return mask


    def self_blending_custom(self,filename_frame,landmark,frame,method='components'):
        landmark_name = filename_frame.replace('png', 'npy').replace('frame', 'landmarks')
        if os.path.exists(landmark_name):
            landmark = np.load(landmark_name)[0]
            landmark=self.reorder_landmark(landmark)
            img,img_blended,mask=self.self_blending(frame.copy(),landmark.copy(),method)
        else:
            pass
        return img,img_blended,mask

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                    # alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                    alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                    alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                ],p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],p=1),
            
        ], p=1.)


    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']

        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask

    def self_blending_mask(self,img,landmark):
        
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask=random_get_hull(landmark,img)[:,:,0]
            logging.disable(logging.NOTSET)
        else:
            mask=np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
        return mask

    def self_blending(self,img,landmark,method='components'):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask=random_get_hull_custom(landmark,img,method)[:,:,0]
            logging.disable(logging.NOTSET)
        else:
            mask=np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
        source = img.copy()
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']
        if method == 'SBI':
            source, mask = self.randaffine(source,mask)
            img_blended,mask=B.dynamic_blend_custom(source,img,mask)
        elif method == 'components':
            img_blended,mask=B.dynamic_blend_custom(source,img,mask)
        else:
            img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)
        return img,img_blended,mask

    # def self_blending_by_mask(self,img,mask,random_value=0):
    #     H,W=len(img),len(img[0])
    #     source = img.copy()
    #     if random_value<0.5:
    #         source = self.source_transforms(image=source.astype(np.uint8))['image']
    #     else:
    #         img = self.source_transforms(image=img.astype(np.uint8))['image']
    #     img_blended,mask=B.dynamic_blend_custom(source,img,mask)
    #     img_blended = img_blended.astype(np.uint8)
    #     img = img.astype(np.uint8)
    #     return img,img_blended,mask


    def self_blending_by_mask(self, img, mask, random_value=0):
        H, W = img.shape[:2]
        is_single_channel = False
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            is_single_channel = True
        elif img.shape[-1] == 1:
            is_single_channel = True

        source = img.copy()
        def apply_transforms(image_in):
            if is_single_channel:
                image_3c = cv2.cvtColor(image_in, cv2.COLOR_GRAY2RGB)
                aug_3c = self.source_transforms(image=image_3c.astype(np.uint8))['image']
                aug_gray = cv2.cvtColor(aug_3c, cv2.COLOR_RGB2GRAY)
                return np.expand_dims(aug_gray, axis=-1)
            else:
                return self.source_transforms(image=image_in.astype(np.uint8))['image']
        if random_value < 0.5:
            source = apply_transforms(source)
        else:
            img = apply_transforms(img)
        img_blended, mask = B.dynamic_blend_custom(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)
        return img, img_blended, mask



    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]
        landmark=landmark.copy()
        bbox=bbox.copy()

        if landmark is not None:
            landmark_new=np.zeros_like(landmark)

            
            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new
    
    def _oversample(self, data, count=-1):
        real=list(filter(lambda x:x[1]==0,data))
        fakes=list(filter(lambda x:x[1]==1,data))
        if count == -1:
            num_real = len(real)
            # if self.mode == "train":
            fakes = random.sample(fakes, num_real)
            # fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        else:
             real = random.sample(real, count)
             fakes = random.sample(fakes, count)
            # fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        return real+fakes
    def collate_fn(self,batch):
        process_imgs,video_labels,filename_frame,sampled_frame_idxs,mask_f,mask_BEiT=zip(*batch)
        data={}
        data['images']=torch.stack(process_imgs)
        data['labels']=torch.stack(video_labels)
        data['video_path'] = filename_frame
        data['sampled_frame_idxs'] = sampled_frame_idxs
        data['masks_f'] = mask_f
        data['mask'] = torch.tensor(mask_BEiT)
        return data
    from collections import namedtuple
