CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11119 train2lossSBI.py -c configs/Stage1.yaml
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11119 train_s2.py -c configs/Stage2.yaml


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12533 myeval.py -c checkpoints/4-29-FFFinalVersion-10Epoch-v4_FinalModel_v1/Stage2_5.yaml

