import math
import torch
from torch import nn
from torch.nn import functional as F
from mmaction.models.backbones import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm
from models.lib.BEiT_v2 import modeling_finetune



def compute_bce_loss(pred_scores, label_scores):
    pred_probs = torch.sigmoid(pred_scores)
    bce_loss = F.binary_cross_entropy(pred_probs, label_scores)
    return bce_loss



class BEiT_v2(nn.Module):
    def __init__(self, 
        pretrained=True, 
        **kwargs):
        super(BEiT_v2,self).__init__()
        feature_dim = 768
        self.backbone_model = modeling_finetune.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model.head = nn.Identity()
        if pretrained:
            checkpoint = torch.load('BEiT_Face750w.tar', map_location='cpu')
            self.backbone_model.load_state_dict(checkpoint['model'], strict=False)
        self.norm_for_cls = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, 2))
    def forward(self, vid_fea, rgb_img, is_eval=False):
        B, N, T, _, _, _ = rgb_img.size()
        x = rgb_img.flatten(0,2)
        ffd_fea = self.backbone_model(x,return_all_tokens=True)[:,0]
        output = self.norm_for_cls(ffd_fea)
        output = self.header(output)
        output = output.view(B,N*T,-1)
        cos_sim = torch.tensor(0).cuda()
        if not is_eval:
            return output,cos_sim
        else:
            return output



class GroupedInteractionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class IntraGroupBlock(nn.Module):
    """
    带 LayerNorm 和残差连接的组内注意力模块。
    保证特征强化过程中不会发生梯度消失或数值爆炸。
    """
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

    def forward(self, x):
        # x shape: [B, Seq_len, Dim]
        # 预归一化 (Pre-Norm) + 注意力 + 残差连接
        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        return x + attn_out


BEiT_Config = {
    'drop_path_rate': 0.1,
    'use_mean_pooling': False,
    'init_values': 0.1,
    'qkv_bias': True,
    'use_abs_pos_emb': False,
    'use_rel_pos_bias': True,
    'use_shared_rel_pos_bias': False
}






            
class FinalModel_v1(nn.Module):
    def __init__(self, **kwargs):
        super(FinalModel_v1, self).__init__()
        feature_dim = 768
        self.backbone_model = modeling_finetune.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model.head = nn.Identity()
        checkpoint = torch.load('BEiT_Face750w.tar', map_location='cpu')
        self.backbone_model.load_state_dict(checkpoint['model'], strict=False)
        self.tsn_model = resnet.ResNet(depth=50, in_channels=10)
        ckpt_tsn = torch.load('tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth', map_location="cpu")
        new_state_dict = {k[9:] if k.startswith('backbone.') else k: v for k, v in ckpt_tsn['state_dict'].items()}
        self.tsn_model.load_state_dict(new_state_dict, strict=False)
        self.flow_proj = nn.Sequential(nn.Conv2d(2048, 768, kernel_size=1),nn.BatchNorm2d(768),nn.ReLU())
        self.flow_patch_classifier = nn.Sequential(nn.Conv2d(768, 384, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(384, 192, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(192, 1, kernel_size=1),)
        self.norm_for_cls = nn.LayerNorm(feature_dim)
        self.header = nn.Sequential(nn.Linear(feature_dim, 2))
        self.num = 0
        
        self.tsn_model.eval()
        self.flow_proj.eval()
        self.backbone_model.eval()
        self.flow_patch_classifier.eval()

        for name, param in self.tsn_model.named_parameters():
            param.requires_grad = False
        for name, param in self.backbone_model.named_parameters():
            param.requires_grad = False
        for name, param in self.flow_proj.named_parameters():
            param.requires_grad = False
        for name, param in self.flow_patch_classifier.named_parameters():
            param.requires_grad = False


        self.motion_proj = nn.Linear(768, 768)
        self.motion_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True, dropout=0.1) 
            for i in range(2)
        ])
        self.scales = [2, 4] # 局部尺度
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True, dropout=0.1) 
            for i in range(2)
        ])

        self.window_sizes = [4, 8, 12, 16] 
        self.tokens_per_frame = [4, 3, 2, 1] 
        num_cross_layers = len(self.window_sizes) 
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True, dropout=0.1)
            for _ in range(num_cross_layers)  
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(768) for _ in range(num_cross_layers)
        ])



        self.fusion_proj = nn.Sequential(nn.Linear(768 * 2, 768),nn.LayerNorm(768))
        self.self_attn_layers_final = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True, dropout=0.1) 
            for i in range(2)
        ])
        self.intra_group_attn = IntraGroupBlock(dim=768, num_heads=12)



    def forward(self, flw_img, rgb_img, flw_img_aug=None, rgb_img_aug=None, region_labels=None, is_eval=False, label=None):
        self.tsn_model.eval()
        self.flow_proj.eval()
        self.backbone_model.eval()
        self.flow_patch_classifier.eval()
        
        with torch.no_grad():
            B, N, T_r, C_r, H, W = rgb_img.size()
            B, N, T_f, C_f, H, W = flw_img.size()
            flw_img = rearrange(flw_img, 'b n t c h w -> (b n t) c h w').float()
            feat_flow_raw = self.tsn_model.forward(flw_img)
            feat_flow_grouped = rearrange(feat_flow_raw, '(b n t) c h w -> (b n) t c h w', b=B, n=N, t=T_f)
            feat_flow_diff = (feat_flow_grouped[:, 2:] - feat_flow_grouped[:, 1:-1]) + (feat_flow_grouped[:, 1:-1] - feat_flow_grouped[:, 0:-2])
            feat_flow_ori = feat_flow_diff
            B_N, T, _, H_f, W_f = feat_flow_ori.shape
            feat_flow_all_flat = rearrange(feat_flow_ori, 'b_n t c h w -> (b_n t) c h w')
            feat_flow_proj = self.flow_proj(feat_flow_all_flat)  # [(B*N)*T, 768, H, W]
            feat_flow_upsampled = F.interpolate(feat_flow_proj, size=(14, 14), mode='bilinear', align_corners=False)
            importance_maps_ori = self.flow_patch_classifier(feat_flow_upsampled)  # [(B*N)*T, 1, 14, 14]
            importance_maps_ori = rearrange(importance_maps_ori, '(b_n t) c h w -> b_n t (h w) c', b_n=B*N, t=4)
            importance_scores_ori = importance_maps_ori.squeeze(-1)
            flow_pred_scores_ori = importance_scores_ori.clone()
            x = rgb_img.flatten(0,2)
            ffd_fea = self.backbone_model(x,return_all_tokens=True, extra_attn_bias=flow_pred_scores_ori)

        cls_tokens   = ffd_fea[:, 0:1]       # [B*N*4, 768]
        patch_tokens = ffd_fea[:, 1:]    # [B*N*4, 196, 768]
        flow_scores_flat = flow_pred_scores_ori.flatten(0, 1) # [B*N*4, 196]
        sorted_scores, sorted_indices = torch.sort(flow_scores_flat, dim=-1, descending=True)
        gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        sorted_patch_tokens = torch.gather(patch_tokens, dim=1, index=gather_indices)
        group_1 = torch.cat([cls_tokens, sorted_patch_tokens[:, 0:49, :]], dim=1)
        group_2 = torch.cat([cls_tokens, sorted_patch_tokens[:, 49:98, :]], dim=1)
        group_3 = torch.cat([cls_tokens, sorted_patch_tokens[:, 98:147, :]], dim=1)
        group_4 = torch.cat([cls_tokens, sorted_patch_tokens[:, 147:196, :]], dim=1)

        group_1 = rearrange(group_1, '(b n t) p c ->b (n t p) c',b=B, n=N, t=T_r)
        group_2 = rearrange(group_2, '(b n t) p c ->b (n t p) c',b=B, n=N, t=T_r)
        group_3 = rearrange(group_3, '(b n t) p c ->b (n t p) c',b=B, n=N, t=T_r)
        group_4 = rearrange(group_4, '(b n t) p c ->b (n t p) c',b=B, n=N, t=T_r)

        g1_out = self.intra_group_attn(group_1)
        g2_out = self.intra_group_attn(group_2)
        g3_out = self.intra_group_attn(group_3)
        g4_out = self.intra_group_attn(group_4)

        g1_out = rearrange(g1_out, 'b (n t p) c ->(b n t) p c',b=B, n=N, t=T_r)[:,0:1]
        g2_out = rearrange(g2_out, 'b (n t p) c ->(b n t) p c',b=B, n=N, t=T_r)[:,0:1]
        g3_out = rearrange(g3_out, 'b (n t p) c ->(b n t) p c',b=B, n=N, t=T_r)[:,0:1]
        g4_out = rearrange(g4_out, 'b (n t p) c ->(b n t) p c',b=B, n=N, t=T_r)[:,0:1]


        ffd_fea = torch.cat([cls_tokens, g1_out, g2_out, g3_out, g4_out], dim=1)
        ffd_fea = torch.mean(ffd_fea, dim=1)
        frame_multi_tokens = torch.cat([g1_out, g2_out, g3_out, g4_out], dim=1)
        '''Stage2-Step1:光流运动建模'''
        feat_flow_pooled = F.adaptive_avg_pool2d(feat_flow_proj, (1, 1)).squeeze(-1).squeeze(-1) # [(B*N*4), 768]
        feat_flow_seq = rearrange(feat_flow_pooled, '(b n t) c -> b n t c', b=B, n=N, t=4)
        clip_motion_feat = feat_flow_seq.mean(dim=2)
        # clip_motion_feat = clip_motion_feat + self.motion_pos_embed

        for motion_attn_layer in self.motion_attn_layers: 
            clip_motion_feat, _ = motion_attn_layer(query=clip_motion_feat, key=clip_motion_feat, value=clip_motion_feat, need_weights=False)

        modeled_motion_expanded = clip_motion_feat.unsqueeze(2).expand(-1, -1, 4, -1)
        modeled_motion_flat = rearrange(modeled_motion_expanded, 'b n t c -> b (n t) c')
        modeled_motion_aligned = self.motion_proj(modeled_motion_flat)
        

        '''Stage2-Step2:预测与一致性感知的融合模块'''
        ffd_fea_seq = rearrange(ffd_fea, '(b n t) c -> b (n t) c', b=B, n=N, t=4) # [B, N*4, 768]
        cross_out = modeled_motion_aligned
        
        kv_seq = rearrange(frame_multi_tokens, '(b n_t) k c -> b n_t k c', b=B)

        for s_target, t_count, cross_attn_layer, layer_norm in zip(self.window_sizes, self.tokens_per_frame, self.cross_attn_layers, self.layer_norms):
            seq_len = cross_out.size(1) 
            s = s_target if seq_len % s_target == 0 else seq_len 
            if seq_len % s != 0:
                s = seq_len
            q_window = rearrange(cross_out, 'b (w s) c -> (b w) s c', s=s)
            current_kv = kv_seq[:, :, :t_count, :] # 截取前 t_count 个分组 [B, N*4, t_count, 768]
            k_window = rearrange(current_kv, 'b (w s) t c -> (b w) (s t) c', s=s)
            v_window = k_window
            attn_out, _ = cross_attn_layer(query=q_window, key=k_window, value=v_window, need_weights=False)
            attn_out = rearrange(attn_out, '(b w) s c -> b (w s) c', b=B, s=s)
            cross_out = layer_norm(cross_out + attn_out)
        
        concat_feat = torch.cat([ffd_fea_seq, cross_out], dim=-1)
        ffd_fea_amplified = self.fusion_proj(concat_feat)

        '''Stage2-Step3:RGB强化后的特征进行时序建模'''
        for attn_layer in self.self_attn_layers: 
            ffd_fea_amplified, _ = attn_layer(query=ffd_fea_amplified, key=ffd_fea_amplified, value=ffd_fea_amplified, need_weights=False)



        ffd_fea_out = ffd_fea_seq+ffd_fea_amplified
        for attn_layer in self.self_attn_layers_final:
            ffd_fea_out, _ = attn_layer(query=ffd_fea_out, key=ffd_fea_out, value=ffd_fea_out, need_weights=False)


        ffd_fea_out = (ffd_fea_seq+ffd_fea_out).flatten(0, 1)
        output = self.norm_for_cls(ffd_fea_out)
        output = self.header(output)
        output = output.view(B, N*4, -1)
        
        total_loss = torch.tensor(0.0, device=output.device)
        if is_eval:
            return output
        else:
            return [output, total_loss]
    

