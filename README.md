## Flow-Guided Face Forgery Detection (FG-FFD)

[Introduction](#introduction) |
[Preparation](#Preparation) |
[Get Started](#get-started) |


### Introduction

Face Forgery Detection (FFD) struggles with cross-dataset generalization due to forgery distribution shifts between training and unseen manipulation algorithms. Existing FFD methods mitigate this generalization gap primarily through data augmentation, spatial representation, and spatio-temporal modeling. To exploit inherent spatio-temporal disruptions of facial manipulations, we seek to leverage optical flow variations as an explicit guiding signal, unifying these three perspectives into a comprehensive FFD pipeline. Accordingly, we propose Flow-Guided Face Forgery Detection (FG-FFD) framework, comprising three synergistic modules to progressively unravel spatio-temporal inconsistencies. Specifically, Variation-Guided Augmentation leverages adjacent-frame flow variation intensities to selectively augment specific facial component regions. Subsequently, Region-Guided Anomaly Identification utilizes flow-predicted patch-wise probabilities to direct spatial attention toward suspected anomalies. Finally, Hierarchy-Guided Inconsistency Aggregation facilitates multi-level and multi-scale temporal interactions to unravel spatio-temporal cues. Extensive experiments demonstrate that FG-FFD achieves state-of-the-art generalization across diverse cross-dataset and cross-manipulation scenarios.

### Preparation

#### 1. Environment and Dependencies:

This project is implemented with Python version >= 3.10 and CUDA version >= 11.3.

It is recommended to follow the steps below to configure the environment:
```
conda create -n FG-FFD python=3.10
conda activate FG-FFD
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

#### 2.Data Preparation:

Before training, follow the steps below to prepare the data:
1. Download datasets.
   
2. Frame Extraction: Extract frames from the video files.
   
3. Face Alignment and Cropping: Referring to the [TFCU](https://github.com/zhenglab/TFCU), RetinaFace was chosen for facial recognition, followed by cropping and alignment procedures. When multiple faces appear in the video, tracking the face with the longest appearance time for preservation. The extraction method of optical flow is referred to as [mmaction==0.24.1](https://github.com/open-mmlab/mmaction2).

### Quickly Inference

Download weights from [Baidu Cloud(code: ffvd)](https://pan.baidu.com/s/1uZIQcYTlZermQ4HhjfepNA) and put it into 'checkpoints/Stage2_FinalModel_v1/ckpt'.

Infer a single video: Run the ```python inference.py```.

###  Evaluation

Download weights 'Final.tar' from [Baidu Cloud(code: ffvd)](https://pan.baidu.com/s/1uZIQcYTlZermQ4HhjfepNA) and put it into 'checkpoints/Stage2_FinalModel_v1' . Then run:

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12533 myeval.py -c checkpoints/Stage2_FinalModel_v1
```

###  Train
Download pretrained weights 'BEiT_Face750w.tar' from [Baidu Cloud(code: ffvd)](https://pan.baidu.com/s/1uZIQcYTlZermQ4HhjfepNA). Then run:
```
#Stage1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11119 train2lossSBI.py -c configs/Stage1.yaml
#Stage2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11119 train_s2.py -c configs/Stage2.yaml
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Celeb-DF</th>
<th valign="bottom">DFDC</th>
<th valign="bottom">FFIW</th>
<th valign="bottom">Checkpoints</th>
<tr><td align="left">Ours</td>
<td align="center">95.81%</td>
<td align="center">89.55%</td>
<td align="center">92.82%</td>
<td align="center"><a href="https://pan.baidu.com/s/1uZIQcYTlZermQ4HhjfepNA">Baidu(code: ffvd)</a></td>
</tbody></table>


