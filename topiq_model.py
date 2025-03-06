import copy
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import OrderedDict
from typing import Union, List, Dict, Optional
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import clip
except ImportError:
    print("CLIP not installed. Some functionality may be limited.")

# Constants for image normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Pre-trained model URLs - replace with your own hosting or method to distribute weights
MODEL_URLS = {
    'cfanet_nr_koniq_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_koniq_res50-3de71bal.pth',
    'cfanet_fr_kadid_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_fr_kadid_res50-c6e5a1aa.pth',
    'cfanet_nr_flive_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_flive_res50-0d9c8926.pth',
    'cfanet_nr_spaq_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_spaq_res50-a7f799ac.pth',
    'cfanet_iaa_ava_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_iaa_ava_res50-3cd62bb3.pth',
    'cfanet_iaa_ava_swin': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_iaa_ava_swin-393b41b4.pth',
    'topiq_nr_gfiqa_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/topiq_nr_gfiqa_res50-d76bf1ae.pth',
    'topiq_nr_cgfiqa_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/topiq_nr_cgfiqa_res50-0a8b8e4f.pth',
    'topiq_nr_cgfiqa_swin': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/topiq_nr_gfiqa_swin-7bb80a60.pth',
}

# TOPIQ model configurations
TOPIQ_CONFIGS = {
    'topiq_nr': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_koniq_res50',
            'use_ref': False,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_nr-flive': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_flive_res50',
            'use_ref': False,
            'test_img_size': 384,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_nr-spaq': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_spaq_res50',
            'use_ref': False,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_nr-face': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'topiq_nr_cgfiqa_res50',
            'use_ref': False,
            'test_img_size': 512,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_nr_swin-face': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'swin_base_patch4_window12_384',
            'model_name': 'topiq_nr_cgfiqa_swin',
            'use_ref': False,
            'test_img_size': 384,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_nr-face-v1': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'topiq_nr_gfiqa_res50',
            'use_ref': False,
            'test_img_size': 512,
        },
        'metric_mode': 'NR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_fr': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_fr_kadid_res50',
            'use_ref': True,
        },
        'metric_mode': 'FR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_fr-pipal': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_fr_pipal_res50',
            'use_ref': True,
        },
        'metric_mode': 'FR',
        'score_range': '~0, ~1',
        'lower_better': False,
    },
    'topiq_iaa': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'swin_base_patch4_window12_384',
            'model_name': 'cfanet_iaa_ava_swin',
            'use_ref': False,
            'inter_dim': 512,
            'num_heads': 8,
            'num_class': 10,
        },
        'metric_mode': 'NR',
        'score_range': '1, 10',
        'lower_better': False,
    },
    'topiq_iaa_res50': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_iaa_ava_res50',
            'use_ref': False,
            'inter_dim': 512,
            'num_heads': 8,
            'num_class': 10,
        },
        'metric_mode': 'NR',
        'score_range': '1, 10',
        'lower_better': False,
    },
}

# CLIP model loading utility
def load_clip_model(name, device="cpu", jit=False, download_root=None):
    """Load a CLIP model"""
    if hasattr(clip, 'load'):
        return clip.load(name, device=device, jit=jit, download_root=download_root)
    else:
        raise ImportError("CLIP module is not properly installed")

def download_pretrained_model(url, save_path):
    """Download pretrained model from url"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        import requests
        print(f'Downloading model from {url} to {save_path}')
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def load_pretrained_network(model, state_dict_path, strict=True):
    """Load pretrained model weights"""
    if not os.path.exists(state_dict_path):
        # Try to download if it's a known model
        for model_name, url in MODEL_URLS.items():
            if model_name in state_dict_path:
                os.makedirs(os.path.dirname(state_dict_path), exist_ok=True)
                download_success = download_pretrained_model(url, state_dict_path)
                if not download_success:
                    print(f"Could not download model {model_name}. Please download manually.")
                    return model
                break

    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        if 'params' in state_dict:
            state_dict = state_dict['params']
        
        # Remove module. prefix
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # remove 'module.'
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict, strict=strict)
        print(f"Successfully loaded pretrained model from {state_dict_path}")
    else:
        print(f"Cannot load model from {state_dict_path}, file does not exist")
    
    return model

def load_image(image_path):
    """Load an image and convert to tensor"""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img
    return image_path  # Assume already a tensor

# Helper functions for transformer architecture
def _get_clones(module, N):
    """Create N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor):
        return tensor

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor):
        return tensor

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2,
                                                  key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output


class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.conv(x) * self.sig(self.gate(x))


# Wrapper function for creating Swin models
def create_swin(name, pretrained=True, **kwargs):
    """Create Swin Transformer model using timm"""
    if not name.startswith('swin_'):
        # Ensure proper naming convention
        if name == 'swin_base_patch4_window12_384':
            pass  # Name is already correct
        elif 'swin_base' in name and '384' in name:
            name = 'swin_base_patch4_window12_384'
        else:
            raise ValueError(f"Unsupported Swin model name: {name}")
    
    try:
        model = timm.create_model(name, pretrained=pretrained, **kwargs)
        # Add feature_dim_list attribute needed by TOPIQ
        if not hasattr(model, 'feature_dim_list'):
            # For typical Swin models, feature dimensions depend on architecture
            if 'base' in name:
                model.feature_dim_list = [128, 256, 512, 1024]
            elif 'large' in name:
                model.feature_dim_list = [192, 384, 768, 1536]
            elif 'tiny' in name:
                model.feature_dim_list = [96, 192, 384, 768]
            else:
                model.feature_dim_list = [96, 192, 384, 768]  # Default to tiny
        return model
    except Exception as e:
        print(f"Error creating Swin model: {e}")
        raise


# Main TOPIQ model implementation
class CFANet(nn.Module):
    """
    Cross-scale Feature Attention Network (CFANet) - Core of the TOPIQ model.
    
    Args:
        semantic_model_name (str): Backbone network for feature extraction ('resnet50' or 'swin_base_patch4_window12_384').
        model_name (str): Name of the specific model to load weights for.
        backbone_pretrain (bool): Whether to load pretrained weights for backbone.
        in_size (int): Input size to resize images to. If None, uses original size.
        use_ref (bool): Whether to use reference images (FR mode) or not (NR mode).
        num_class (int): Number of classes for output. For quality assessment, typically 1.
        num_crop (int): Number of crops to use for inference.
        crop_size (int): Size of each crop.
        inter_dim (int): Dimension of intermediate features.
        num_heads (int): Number of attention heads.
        num_attn_layers (int): Number of attention layers.
        dprate (float): Dropout rate.
        activation (str): Activation function ('relu' or 'gelu').
        pretrained (bool): Whether to load pretrained weights.
        pretrained_model_path (str): Path to pretrained weights.
        out_act (bool): Whether to use activation on output.
        block_pool (str): Method for block pooling ('avg', 'weighted_avg').
        test_img_size (int): Image size for testing. If None, uses crop_size.
        align_crop_face (bool): Whether to crop and align faces for face IQA.
    """
    def __init__(self,
                 semantic_model_name='resnet50',
                 model_name='cfanet_nr_koniq_res50',
                 backbone_pretrain=True,
                 in_size=None,
                 use_ref=False,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=True,
                 pretrained_model_path=None,
                 out_act=False,
                 block_pool='weighted_avg',
                 test_img_size=None,
                 align_crop_face=True,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD
                 ):
        super(CFANet, self).__init__()
        # Save configuration
        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.num_class = num_class
        self.use_ref = use_ref
        self.semantic_level = -1  # Use final layer features
        self.in_size = in_size
        self.num_crop = num_crop
        self.crop_size = crop_size
        self.test_img_size = test_img_size if test_img_size is not None else crop_size
        self.block_pool = block_pool
        self.align_crop_face = align_crop_face

        # Load backbone model (ResNet or Swin Transformer)
        if 'swin' in semantic_model_name.lower():
            self.semantic_model = create_swin(semantic_model_name, pretrained=backbone_pretrain)
            feature_dim_list = self.semantic_model.feature_dim_list
            default_mean, default_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # Swin uses ImageNet normalization
        elif 'clip' in semantic_model_name.lower():
            semantic_model_name = semantic_model_name.replace('clip_', '')
            self.semantic_model = [load_clip_model(semantic_model_name, 'cpu')]
            feature_dim_list = self.semantic_model[0].visual.feature_dim_list if hasattr(self.semantic_model[0].visual, 'feature_dim_list') else [768]  # Default for CLIP
            default_mean, default_std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        else:
            # Use ResNet by default
            self.semantic_model = timm.create_model(semantic_model_name, pretrained=backbone_pretrain, features_only=True)
            feature_dim_list = self.semantic_model.feature_info.channels()
            feature_dim = feature_dim_list[self.semantic_level]
            all_feature_dim = sum(feature_dim_list)
            self.fix_bn(self.semantic_model)
        
        # Set normalization constants
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # Define cross-attention transformer architecture
        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # Cross-attention pooling
        self.h_emb = nn.Parameter(torch.zeros(1, inter_dim // 2, feature_dim_list[-1], 1))
        self.w_emb = nn.Parameter(torch.zeros(1, inter_dim // 2, 1, feature_dim_list[-1]))
        nn.init.normal_(self.h_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.w_emb, mean=0.0, std=0.02)

        # Self-attention blocks for each scale
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, 
                                          dim_feedforward=dim_feedforward, 
                                          normalize_before=True, dropout=dprate, 
                                          activation=activation)
        
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                # For FR mode, create weighted pooling modules
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1),
                        self.act_layer,
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer,
                        nn.Conv2d(64, 1, 3, stride=1, padding=1),
                        nn.Sigmoid()
                    )
                )
            else:
                # For NR mode, use gated convolution
                self.weight_pool.append(GatedConv(dim, dim))
                
            # Dimensionality reduction for each feature map
            self.dim_reduce.append(nn.Sequential(
                nn.Conv2d(dim, inter_dim, 1, 1),
                self.act_layer,
                )
            )

            # Self-attention blocks for each scale
            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # Cross-scale attention blocks
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(inter_dim, nhead=num_heads, 
                                          dim_feedforward=dim_feedforward, 
                                          normalize_before=True, dropout=dprate, 
                                          activation=activation)
        
        # Create cross-scale attention blocks
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # Attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(inter_dim, nhead=num_heads, 
                                               dim_feedforward=dim_feedforward, 
                                               normalize_before=True, dropout=dprate, 
                                               activation=activation)

        # Output MLP layers
        linear_dim = inter_dim
        score_linear_layers = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]
        
        if out_act:
            score_linear_layers.append(nn.Sigmoid())
        
        # Create the final score prediction module
        self.score_predictor = nn.Sequential(*score_linear_layers)
        
        # Apply Xavier/Kaiming initialization to linear layers
        self.apply(self._init_linear)
        
        # Load pretrained weights if available
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True)
        elif pretrained:
            model_path = os.path.join(os.path.expanduser('~'), '.cache', 'topiq', model_name + '.pth')
            load_pretrained_network(self, model_path, True)

    def _init_linear(self, m):
        """Initialize linear layers"""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def preprocess(self, x):
        """Preprocess input images"""
        if x.shape[-1] != self.test_img_size or x.shape[-2] != self.test_img_size:
            x = F.interpolate(x, size=(self.test_img_size, self.test_img_size), mode='bicubic')
        return x

    def fix_bn(self, model):
        """Set batch normalization layers to eval mode"""
        if isinstance(model, nn.BatchNorm2d):
            model.eval()
        for name, child in model.named_children():
            self.fix_bn(child)

    def dist_func(self, x, y, eps=1e-12):
        """Calculate distance between features"""
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x, y=None):
        """
        Forward pass through cross-attention network
        
        Args:
            x (torch.Tensor): Distorted image/features
            y (torch.Tensor, optional): Reference image/features
        
        Returns:
            torch.Tensor: Quality score
        """
        # Process through backbone model
        if 'clip' in self.semantic_model_name.lower():
            visual_model = self.semantic_model[0].visual.to(x.device)
            dist_feat_list = visual_model.forward_features(x)
            if self.use_ref:
                ref_feat_list = visual_model.forward_features(y)
        else:
            dist_feat_list = self.semantic_model(x)
            if self.use_ref:
                ref_feat_list = self.semantic_model(y)
            self.fix_bn(self.semantic_model)
            self.semantic_model.eval()
        
        # Define feature levels
        start_level = 0
        end_level = len(dist_feat_list)

        # Create positional embeddings
        b, c, th, tw = dist_feat_list[-1].shape
        pos_emb = torch.cat((
            self.h_emb.to(x.device).repeat(1, 1, 1, tw), 
            self.w_emb.to(x.device).repeat(1, 1, th, 1)
        ), dim=1)
        
        # Feature tokenization and fusion
        token_list = []
        attn_list = []
        
        for i in range(start_level, end_level):
            dist_feat = dist_feat_list[i]
            
            # Process reference features if available
            if self.use_ref:
                ref_feat = ref_feat_list[i]
                # Calculate distance/similarity map between dist and ref
                dist_map = self.dist_func(dist_feat, ref_feat)
                # Concatenate features
                feat = torch.cat([dist_feat, ref_feat, dist_map], dim=1)
                # Apply weighted pooling
                feat_weight = self.weight_pool[i](dist_map)
                feat = feat * feat_weight
            else:
                feat = self.weight_pool[i](dist_feat)
                
            # Reduce dimensions
            feat_token = self.dim_reduce[i](feat)
            b, c, h, w = feat_token.shape
            
            # Reshape for transformer
            feat_token = feat_token.flatten(2).permute(0, 2, 1)  # B, N, C
            
            # Apply self-attention
            feat_token = self.sa_attn_blks[i](feat_token)
            token_list.append(feat_token)
        
        # Create positional embedding for cross-attention
        pos_embed = pos_emb.flatten(2).permute(0, 2, 1)  # B, N, C
        
        # Apply cross-scale attention
        deep_token = token_list[-1]
        for idx, token in enumerate(reversed(token_list[:-1])):
            # Cross attention between current and deeper layer
            deep_token = self.attn_blks[end_level - idx - 2](token, deep_token)
            
        # Apply final cross attention with positional embedding
        deep_token = self.attn_pool(deep_token + pos_embed)
        
        # Global pooling
        if self.block_pool == 'avg':
            deep_token = torch.mean(deep_token, dim=1)
        elif self.block_pool == 'max':
            deep_token = torch.max(deep_token, dim=1)[0]
        else:  # weighted_avg
            deep_token = torch.mean(deep_token, dim=1)
        
        # Predict score
        score = self.score_predictor(deep_token)
        
        return score

    def forward(self, x, y=None, return_mos=True, return_dist=False):
        """
        Forward pass through TOPIQ model
        
        Args:
            x (torch.Tensor): Distorted image
            y (torch.Tensor, optional): Reference image
            return_mos (bool): Whether to return MOS score
            return_dist (bool): Whether to return distribution (for IAA)
            
        Returns:
            torch.Tensor: Quality scores
        """
        # Prepare device and normalize input
        device = x.device
        self.default_mean = self.default_mean.to(device)
        self.default_std = self.default_std.to(device)
        
        # Preprocess images
        x = self.preprocess(x)
        x = (x - self.default_mean) / self.default_std
        
        if self.use_ref and y is not None:
            # Preprocess reference image
            y = self.preprocess(y)
            y = (y - self.default_mean) / self.default_std
        
        # Forward pass through cross-attention network
        pred = self.forward_cross_attention(x, y)
        
        return_list = []
        
        # Prepare outputs
        if return_mos:
            mos = pred.mean(dim=1) if pred.dim() > 2 else pred
            return_list.append(mos)
        
        if return_dist:
            dist = torch.softmax(pred, dim=1)
            return_list.append(dist)
        
        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


# Create TOPIQ model from configuration
def create_topiq(model_name='topiq_nr', device='cpu', pretrained=True, pretrained_model_path=None, **kwargs):
    """
    Create TOPIQ model from configuration
    
    Args:
        model_name (str): Name of the TOPIQ model variant
        device (str): Device to use ('cpu' or 'cuda')
        pretrained (bool): Whether to load pretrained weights
        pretrained_model_path (str): Path to pretrained weights
        **kwargs: Additional model options
        
    Returns:
        CFANet: TOPIQ model
    """
    if model_name not in TOPIQ_CONFIGS:
        raise ValueError(f"Unknown TOPIQ model: {model_name}, available models: {list(TOPIQ_CONFIGS.keys())}")
    
    # Get model configuration
    config = TOPIQ_CONFIGS[model_name]
    metric_opts = config['metric_opts']
    
    # Override with any provided kwargs
    for k, v in kwargs.items():
        if k in metric_opts:
            metric_opts[k] = v
    
    # Create model
    model = CFANet(
        semantic_model_name=metric_opts['semantic_model_name'],
        model_name=metric_opts['model_name'],
        use_ref=metric_opts['use_ref'],
        pretrained=pretrained,
        pretrained_model_path=pretrained_model_path,
        **{k: v for k, v in metric_opts.items() if k not in ['type', 'semantic_model_name', 'model_name', 'use_ref']}
    )
    
    # Move to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model