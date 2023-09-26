############################################################################################################

# Modified The Original AST code from:

# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

############################################################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import wget
import numpy as np
import timm
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from typing import List, Optional
import math
from src.utilities.resizing_helper import divs, gcd, resample_abs_pos_embed, resample_patch_embed, vanilla_resample_patch_embed, get_resize_mat_pinv
from src.models.ast_models import ASTModel
import random

# TODO: Right now, only support square patch size! Later on, extend to rectangular patch size too
class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16), # TODO: Support rectangular patch sizes too
        in_chans=1, # Because it is audio
        embed_dim=768,
        bias=True, # TODO: Later, handle the False case
        norm_layer=None,
        flatten=True,
        proj_load=None,
        resize_func=resample_patch_embed,
        precompute_for=None,
        verbose=True
    ):
        super().__init__()

        print(f'Resize function is {resize_func.__name__}')

        if verbose:
            print(f'Initializing FlexiPatchEmbed with the following parameters:')
            print(f'patch_size={patch_size}, in_chans={in_chans}, embed_dim={embed_dim}, bias={bias}, norm_layer={norm_layer}, flatten={flatten}, proj_load={"yes" if proj_load is not None else None}, resize_func={resize_func.__name__}')
        
        self.patch_size = to_2tuple(patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.resize_func = resize_func

        if verbose:
            print(f'The resize function is {self.resize_func.__name__}')

        if proj_load is not None:

            if verbose:
                print(f'Loading projection weights!')
                print(f'The shapes of the current projection: bias={self.proj.bias.shape}, weight={self.proj.weight.shape}')
                print(f'The shapes of the loaded projection: bias={proj_load.bias.shape}, weight={proj_load.weight.shape}')

            if proj_load.bias.shape != self.proj.bias.shape:
                raise ValueError("The bias shape of the loaded projection layer does not match the current projection layer")
            
            # copy the bias
            self.proj.bias = nn.Parameter(proj_load.bias)

            if proj_load.weight.shape != self.proj.weight.shape:
                self.proj.weight = nn.Parameter(self.resize_func(
                    proj_load.weight,
                    list(self.patch_size)
                ))
                if verbose:
                    print(f'Resized the projection weights with {self.resize_func.__name__}')
                    print(f'The shapes of the resized projection weights={self.proj.weight.shape}')
            else:
                self.proj.weight = nn.Parameter(proj_load.weight)

        self.precomputed_matrices = {}

        if precompute_for is not None:
            if type(precompute_for) is not list:
                raise ValueError("The precompute_for should be either None or a list!")             
            
            if self.resize_func.__name__ != resample_patch_embed.__name__:
                raise ValueError("The precompute_for is only supported when the resize_func is resample_patch_embed!")

            precompute_for = [to_2tuple(patch_size) for patch_size in list(precompute_for)]
            
            for patch_size in precompute_for:
                self.precomputed_matrices[patch_size] = get_resize_mat_pinv(
                    list(self.patch_size),
                    list(patch_size),
                ).detach()
            
            if verbose:
                print(f'Precomputed weights for {precompute_for}')

    def forward(self, x, patch_size=None):
        B, C, H, W = x.shape
        
        if patch_size is None:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        if patch_size == self.patch_size:
            weight = self.proj.weight
        elif patch_size in self.precomputed_matrices:
            weight = self.resize_func(
                self.proj.weight,
                list(patch_size),
                resize_mat_pinv = self.precomputed_matrices[patch_size]
                    .to(x.device)
            )
        else:
            weight = self.resize_func(
                self.proj.weight,
                list(patch_size)
            )
            

        bias = self.proj.bias

        # print(weight.device)

        # TODO: weight's gradients
        x = F.conv2d(x, weight, bias=bias, stride=patch_size)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class FlexiPosEmbed(nn.Module):
    def __init__(
        self,
        input_size=(128, 1024),
        pos_grid_size=(8, 64),
        embed_dim=768,
        pos_embed_load=None,
        pos_grid_size_load=None,
        n_prefix_tokens=1, # Assuming there is a cls token by default
        pos_embed_prefix=True,
        verbose=True
    ):
        super().__init__()

        if verbose:
            print(f'Initializing FlexiPosEmbed with the following parameters:')
            print(f'input_size={input_size}, pos_grid_size={pos_grid_size}, embed_dim={embed_dim}, pos_embed_load={pos_embed_load.shape if pos_embed_load is not None else None}, pos_grid_size_load={pos_grid_size_load}, n_prefix_tokens={n_prefix_tokens}, pos_embed_prefix={pos_embed_prefix}')

        self.input_size = to_2tuple(input_size)
        self.pos_grid_size = to_2tuple(pos_grid_size)

        # TODO: Consider the overflowing patch size case. Either let them overflow, or not
        num_patches = self.pos_grid_size[0] * self.pos_grid_size[1]
        self.n_prefix_tokens = n_prefix_tokens
        self.pos_embed_prefix = pos_embed_prefix
        self.embed_dim = embed_dim
        pos_embed_shape = (1, num_patches + (n_prefix_tokens if pos_embed_prefix else 0), embed_dim)

        if pos_embed_load is not None:

            if verbose:
                print(f'Loading position embedding!')
                print(f'The shape of the current grid size: {pos_grid_size}')
                print(f'The shape of the loaded grid size: {pos_grid_size_load}')

            if pos_grid_size_load is None:
                raise ValueError("The loaded position embedding does not have the grid size information")
            
            if pos_grid_size != pos_grid_size_load:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        pos_embed_load,
                        new_size=self.pos_grid_size,
                        old_size=pos_grid_size_load,
                        num_prefix_tokens=n_prefix_tokens,
                        pos_embed_prefix=self.pos_embed_prefix
                    )
                )
                if verbose:
                    print(f'The shape of the resampled position embedding: {self.pos_embed.shape}')
            else:
                self.pos_embed = nn.Parameter(pos_embed_load)
        else:
            self.pos_embed = nn.Parameter(torch.randn(*pos_embed_shape) * .02) # From the timm implementation
    
    # TODO: Have it support the rectangular patch size too
    # TODO: Update the API to following: get_shape(stride: tuple, patch_size: tuple, input_size: tuple)
    @staticmethod
    def get_shape(fstride, tstride, patch_size, input_fdim, input_tdim):
        patch_size = to_2tuple(patch_size)
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, 1, kernel_size=patch_size, stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x, patch_size):
        old_size = self.pos_grid_size
        patch_size = to_2tuple(patch_size)
        new_size = [*FlexiPosEmbed.get_shape(*patch_size, patch_size, *self.input_size)]
        forward_pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=new_size,
            old_size=old_size, 
            num_prefix_tokens=self.n_prefix_tokens,
            pos_embed_prefix=self.pos_embed_prefix
        )

        if not self.pos_embed_prefix:
            final_patches = (
                x[:, :self.n_prefix_tokens], (x[:, self.n_prefix_tokens:] + forward_pos_embed)
            )
            return torch.cat(final_patches, dim=1)
        
        # print(x, forward_pos_embed)
        # print("x is on:",x.device, "\n pos is on",forward_pos_embed.device)
        
        return x + forward_pos_embed

class FlexiASTModel(nn.Module):
    """
    # TODO: Change the description
    The FlexiAST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(
        self, 
        label_dim=527,
        input_size=(128, 1024),
        orig_patch_size=(16, 16),
        orig_posemb_size=(8, 64),
        embed_dim=None,
        patch_sizes=None,
        only_time=False,
        only_freq=False,
        patch_sizes_prob=None,
        flexivit_pretrain=False,
        ast_pretrain=False,
        ast_params=None,
        imagenet_pretrain=False,
        imagenet_model_name=None,
        load_backbone_only=True,
        model_path=None,
        model_url=None,
        precompute_patch_embed=False,
        verbose=True,
        resize_func=resample_patch_embed
    ):


        super(FlexiASTModel, self).__init__()
        assert timm.__version__ == '0.8.6dev0', 'Please use timm == 0.8.6.dev0 or similar version, the code might not be compatible with older versions.'

        if verbose == True:
            print('---------------FlexiAST Model Summary---------------')
            print(f'AST pretraining: {str(ast_pretrain)}, ImageNet pretraining: {str(imagenet_pretrain)}, FlexiViT pretraining: {str(flexivit_pretrain)}')

        self.label_dim = label_dim
        self.input_size = input_size

        self.orig_patch_size = orig_patch_size
        self.orig_posemb_size = orig_posemb_size
        self.embed_dim = embed_dim

        self.origin = False
        
        self.patch_sizes = patch_sizes if patch_sizes is not None else divs(gcd(*input_size))
        if only_time and only_freq:
            raise ValueError('Only time and only frequency cannot be both True')
        if only_time:
            print(f'Resize pacth embedding ONLY along time axis!')
            self.patch_sizes = [(orig_patch_size[0], x) for x in self.patch_sizes]
        if only_freq:
            print(f'Resize pacth embedding ONLY along frequency axis!')
            self.patch_sizes = [(x, orig_patch_size[1]) for x in self.patch_sizes]
            
        self.patch_sizes_prob = patch_sizes_prob
        
        self.flexivit_pretrain = flexivit_pretrain
        
        self.ast_pretrain = ast_pretrain
        self.ast_params = ast_params

        self.imagenet_pretrain = imagenet_pretrain
        self.imagenet_model_name = imagenet_model_name
        
        self.model_path = model_path
        self.model_url = model_url
        self.load_backbone_only = load_backbone_only
        
        self.precompute_patch_embed = precompute_patch_embed
        self.verbose = verbose
        self.resize_func = resize_func

        if ast_pretrain == True:
            self.load_from_ast_pretrain()
        
        elif imagenet_pretrain == True:
            self.load_from_imagenet_pretrain()
        
        else: 
            raise NotImplementedError('FlexiAST needs to be pretrained with either AST or ImageNet')

    def load_from_ast_pretrain(self):

        if self.model_path is None:
            # raise ValueError('ast_pretrain=True does only support model_path to be specified')
            print("model_path is None. Make sure you are doing evaluation now.")
        else:
            print("Load the pretrained model for training")

        if self.verbose:
            print(f'Using AST pretrained model {self.model_path}')


        ast_model = ASTModel(**self.ast_params)

        if self.model_path is not None:
            state_dict = torch.load(self.model_path, map_location='cpu')
            out_dict = {}
            for k, v in state_dict.items(): # Adjust the name of dict to adapt to nn.DataParallel
                out_dict[k[7:]] = v
            ast_model.load_state_dict(out_dict)

        self.v = ast_model.v
        ast_embed_dim = self.v.pos_embed.shape[2]

        if self.embed_dim is not None and self.embed_dim != ast_embed_dim:
            raise ValueError(f'embed_dim should be {ast_embed_dim} when ast_pretrain=True')
        self.embed_dim = ast_embed_dim

        if self.load_backbone_only:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.label_dim))
        else:
            if self.label_dim != ast_model.mlp_head[-1].out_features:
                raise ValueError(f'label_dim should be {ast_model.mlp_head[-1].out_features} when ast_pretrain=True and load_backbone_only=False')
            self.mlp_head = ast_model.mlp_head

        # NOTE: Normally, ViT-224 is used as our backbone in most experiments.
        # if self.origin True, then the backbone is DeiT-384. This is only for verifying we can also flexify the DeiT model.
        self.origin = (self.ast_params.get('model_size', '224') == 'origin') 

        self.patch_embed = FlexiPatchEmbed(
            patch_size=self.orig_patch_size, # TODO: Support rectangular patch sizes too
            embed_dim=self.embed_dim,
            proj_load=self.v.patch_embed.proj,
            verbose=self.verbose,
            precompute_for=self.patch_sizes if self.precompute_patch_embed else None,
            resize_func=self.resize_func,
        )

        # NOTE: We do not follow what AST does for the loading of the positional embeddings
        self.pos_embed = FlexiPosEmbed(
            input_size=self.input_size,
            pos_grid_size=self.orig_posemb_size,
            embed_dim=self.embed_dim,
            pos_embed_load=self.v.pos_embed,
            pos_grid_size_load=FlexiPosEmbed.get_shape(
                self.ast_params.get('fstride', None),
                self.ast_params.get('tstride', None),
                self.ast_params.get('patch_size', None),
                self.ast_params.get('input_fdim', None),
                self.ast_params.get('input_tdim', None),
            ),
            n_prefix_tokens=(2 if self.origin else 1),
            verbose=self.verbose,
        )


    def load_from_imagenet_pretrain(self):
        """
            Training without ast pretrained weights is just an ablation study.
            We only implemented one setting in the code.
            Please pay attention to the parameters when you want to play with it.
        """
        
        if self.imagenet_model_name is None:
            raise ValueError('imagenet_pretrain=True does only support imagenet_model_name to be specified')
        
        if self.verbose:
            print(f'Using imagenet pretrained model {self.imagenet_model_name}')
    
        # TODO: give some model_name as demo
        self.v = timm.create_model(self.imagenet_model_name, pretrained=True)
    
        v_embed_dim = self.v.pos_embed.shape[2]
        if self.embed_dim is not None and self.embed_dim != v_embed_dim:
            raise ValueError(f'Provided embed_dim argument does not match the model the weights are loaded from. It should be {v_embed_dim}')
        self.embed_dim = v_embed_dim
        
        if self.load_backbone_only == True:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.label_dim))
        else:
            raise NotImplementedError('load_backbone_only=False is not implemented yet for imagenet_pretrain=True')

        # automatcially get the intermediate shape
        # TODO: Adjust the arguments after teh API is updated. Currently, the assumption with the patch size is that it is a square
        f_dim, t_dim = FlexiPosEmbed.get_shape(*self.orig_patch_size, self.orig_patch_size[0], *self.input_size) 
        orig_num_patches = f_dim * t_dim
        
        if self.verbose == True:
            print(f'Original number of patches={orig_num_patches}')

        new_proj = torch.nn.Conv2d(1, self.embed_dim, kernel_size=self.orig_patch_size, stride=self.orig_patch_size)
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias

        vit_num_patches = self.v.patch_embed.num_patches
        vit_hw = int(vit_num_patches ** 0.5)

        # get the positional embedding from vit model, skip the first token (cls token), reshape it to original 2D shape 
        new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, vit_num_patches, self.embed_dim).transpose(1, 2).reshape(1, self.embed_dim, vit_hw, vit_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= vit_hw:
            l = vit_hw//2 - t_dim//2
            r = l + t_dim
            new_pos_embed = new_pos_embed[:, :, :, l : r]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(vit_hw, t_dim), mode='bilinear')

        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= vit_hw:
            l = vit_hw//2 - f_dim//2
            r = l + f_dim
            new_pos_embed = new_pos_embed[:, :, l : r, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.embed_dim, orig_num_patches).transpose(1, 2)
        # concatenate the above positional embedding with the cls token of the vit model.
        new_pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))

        self.patch_embed = FlexiPatchEmbed(
            patch_size=self.orig_patch_size, # TODO: Support rectangular patch sizes too
            embed_dim=self.embed_dim,
            proj_load=new_proj,
            verbose=self.verbose,
            precompute_for=self.patch_sizes if self.precompute_patch_embed else None,
        )

        # TODO: Consider Deit too!!
        self.pos_embed = FlexiPosEmbed(
            input_size=self.input_size,
            pos_grid_size=self.orig_posemb_size, # !!
            embed_dim=self.embed_dim,
            pos_embed_load=new_pos_embed,
            pos_grid_size_load=(f_dim, t_dim),
            n_prefix_tokens=1, # the cls token
            verbose=self.verbose,
        )

    @autocast()
    def forward(self, x, patch_size=-1):
    # def forward(self, x): # Only for FLOP counting
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param patch_size: the patch size to be used while feedforwarding the input spectrogram. 
            If -1, a random patch size is chosen from the list of patch sizes (seqhw). 
            If None, the original patch size is used.
        :return: prediction
        """

        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]

        if patch_size is None:
            patch_size = self.orig_patch_size

        elif patch_size == -1:
            patch_size = random.choice(self.patch_sizes)

        patch_size = to_2tuple(patch_size)
        # if self.only_time:
        # patch_size = to_2tuple(self.orig_patch_size,patch_size[1])

        if self.verbose:
            print(f'patch_size for forward pass: {patch_size}')

        x = self.patch_embed(x, patch_size)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        if self.origin:
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x, patch_size) # !

        x = self.v.pos_drop(x)
        x = self.v.blocks(x)
        x = self.v.norm(x)
        if self.origin:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]

        x = self.mlp_head(x)
        return x

if __name__ == '__main__':

    seq_patch_size = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    input_tdim = 1024
    # original patch size is gonna be 16
    ast_mdl = FlexiASTModel(input_size=(128, input_tdim), patch_sizes=seq_patch_size, imagenet_pretrain=True, imagenet_model_name='vit_base_patch16_224', precompute_patch_embed=True, )
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    
    # test with the original patch size
    test_output = ast_mdl(test_input, patch_size=None)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(f'Test output shape: {test_output.shape}')

    # test with a random patch size
    test_output = ast_mdl(test_input, patch_size=-1)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(f'Test output shape: {test_output.shape}')

"""

        self, 
        label_dim=527,
        input_size=(128, 1024),
        orig_patch_size=(16, 16),
        orig_posemb_size=(8, 64),
        embed_dim=None,
        patch_sizes=None,
        patch_sizes_prob=None,
        flexivit_pretrain=False,
        ast_pretrain=False,
        imagenet_pretrain=False,
        model_path=None,
        model_url=None,
        imagenet_model_name=None,
        verbose=True,
    
"""