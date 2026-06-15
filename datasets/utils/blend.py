# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import cv2
import numpy as np
import scipy as sp
from skimage.measure import label, regionprops
import random
from PIL import Image
import sys
# from scipy.ndimage import elastic_deform
from scipy.ndimage import map_coordinates, gaussian_filter

# 在apply_mask_augmentation中修改位移场生成
  # 2通道位移

# elastic_deform保持原逻辑
def elastic_deform(image, displacement, sigma):
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=-1)
    displacement = gaussian_filter(displacement, sigma=(sigma, sigma, 0))
    h, w = image.shape[:2]
    y, x = np.mgrid[:h, :w]
    coordinates = [y + displacement[..., 0], x + displacement[..., 1]]
    deformed = map_coordinates(image, coordinates, order=1, mode='reflect')
    return deformed[..., np.newaxis]  # 返回 [H,W,1]

def alpha_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured


import cv2
import numpy as np

def sharpen_mask_laplacian(mask):
    # 确保mask为单通道且为浮点型
    mask_float = mask.astype(np.float32)
    if mask_float.ndim == 3:
        mask_float = mask_float.squeeze(axis=-1)
    
    # 拉普拉斯算子锐化
    laplacian = cv2.Laplacian(mask_float, cv2.CV_32F)
    sharpened = mask_float + 0.5 * laplacian  # 系数控制锐化强度
    
    # 限制到[0,1]并恢复形状
    sharpened = np.clip(sharpened, 0, 1)
    return sharpened[..., np.newaxis] if mask.ndim == 3 else sharpened

def sharpen_mask_unsharp(mask, sigma=1.0, strength=1.5):
    mask_float = mask.astype(np.float32)
    if mask_float.ndim == 3:
        mask_float = mask_float.squeeze(axis=-1)
    
    blurred = cv2.GaussianBlur(mask_float, (0, 0), sigma)
    sharpened = mask_float + strength * (mask_float - blurred)
    
    sharpened = np.clip(sharpened, 0, 1)
    return sharpened[..., np.newaxis] if mask.ndim == 3 else sharpened

def sharpen_mask_morphology(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    gradient = dilated - eroded  # 形态学梯度（边缘）
    sharpened = np.clip(mask + gradient, 0, 1)
    return sharpened


def apply_mask_augmentation(mask_blured):
    """对模糊mask应用随机增强"""
    # 随机选择增强方式
    method = np.random.choice(['morph', 'elastic', 'gradient', 'sharpen', 'none'], p=[0.25, 0.25, 0.1, 0.25, 0.15])  # 调整概率
    # method = 'sharpen'
    if method == 'morph':
        # 形态学操作增强
        kernel_size = np.random.randint(3, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if np.random.rand() > 0.5:
            mask_blured = cv2.dilate(mask_blured, kernel)
        else:
            mask_blured = cv2.erode(mask_blured, kernel)
        mask_blured = mask_blured.reshape(mask_blured.shape+(1,))
    elif method == 'elastic':
        # 弹性形变增强
        alpha = np.random.uniform(10, 30)
        sigma = np.random.uniform(5, 10)
        displacement = np.random.randn(*mask_blured.shape[:2], 2) * alpha  # 双通道位移场
        mask_blured = elastic_deform(mask_blured, displacement, sigma=sigma)
    elif method == 'gradient':
        # 添加径向渐变
        y, x = np.ogrid[:224, :224]
        center_y, center_x = np.random.randint(50, 174, 2)
        radius = np.random.randint(50, 100)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        gradient = np.clip(1 - dist/radius, 0, 1)
        mask_blured = np.minimum(mask_blured, gradient[..., np.newaxis])
    elif method == 'sharpen':
        # 随机选择一种锐化方法
        sharpen_method = np.random.choice(['laplacian', 'unsharp'])
        # sharpen_method = 'unsharp'
        if sharpen_method == 'laplacian':
            mask_blured = sharpen_mask_laplacian(mask_blured)
        elif sharpen_method == 'unsharp':
            mask_blured = sharpen_mask_unsharp(mask_blured, sigma=1.0, strength=1.5)
        else:
            mask_blured = sharpen_mask_morphology(mask_blured, kernel_size=3)
            mask_blured = mask_blured.mean(axis=-1, keepdims=True)
        # print(mask_blured.shape, method,sharpen_method)
    return np.clip(mask_blured, 0, 1)

def dynamic_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	blend_list=[0.25,0.5,0.75,1,1,1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio
	# mask_blured = apply_mask_augmentation(mask_blured)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured

def dynamic_blend_custom(source,target,mask):
    mask_blured = get_blend_mask_custom(mask)
    blend_list=[0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured*=blend_ratio
    mask_blured = apply_mask_augmentation(mask_blured)
    # print(mask_blured.shape, source.shape, target.shape)
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
    return img_blended,mask_blured

def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))

def get_dynamic_kernel_size(mask, min_size=3, max_size=26):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return min_size  # 默认返回最小奇数核
    x, y, w, h = cv2.boundingRect(contours[0])
    region_size = max(w, h)
    # 动态计算核大小（区域尺寸的1/5），并限制在[min_size, max_size]之间
    kernel_size = max(min_size, min(max_size, int(region_size / 10)))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 确保奇数

def get_blend_mask_custom(mask):
    num_nonzero = np.count_nonzero(mask)
    H,W=mask.shape
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    kernel_size = get_dynamic_kernel_size(mask)
    kernel_1 = (kernel_size, kernel_size)
    kernel_2 = (max(3, kernel_size // 2 if kernel_size // 2 % 2 == 1 else kernel_size // 2 + 1), 
                    max(3, kernel_size // 2 if kernel_size // 2 % 2 == 1 else kernel_size // 2 + 1))
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    return mask_blured.reshape((mask_blured.shape+(1,)))


# def get_blend_mask_custom(mask, size_h, size_w, size_GB,kernel_1,kernel_2):
# 	H,W=mask.shape
# 	# size_h=np.random.randint(192,257)
# 	# size_w=np.random.randint(192,257)
# 	mask=cv2.resize(mask,(size_w,size_h))
# 	# kernel_1=random.randrange(5,26,2)
# 	kernel_1=(kernel_1,kernel_1)
# 	# kernel_2=random.randrange(5,26,2)
# 	kernel_2=(kernel_2,kernel_2)
	
# 	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
# 	mask_blured = mask_blured/(mask_blured.max())
# 	mask_blured[mask_blured<1]=0
	
# 	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, size_GB)
# 	mask_blured = mask_blured/(mask_blured.max())
# 	mask_blured = cv2.resize(mask_blured,(W,H))
# 	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
	kernel_list=[(11,11),(9,9),(7,7),(5,5),(3,3)]
	blend_list=[0.25,0.5,0.75]
	kernel_idxs=random.choices(range(len(kernel_list)), k=2)
	blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
	mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
	# print(mask_blured.max())
	mask_blured[mask_blured<mask_blured.max()]=0
	mask_blured[mask_blured>0]=1
	# mask_blured = mask
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
	mask_blured = mask_blured/(mask_blured.max())
	return mask_blured.reshape((mask_blured.shape+(1,)))

