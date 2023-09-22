# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,1,2"
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
import copy
from traintest import train, validate
from utilities.resizing_helper import vanilla_resample_patch_embed, resample_patch_embed,resize_weights

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp_dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--skip_norm", default=False, help="Skip input normlization or not")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
# parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

parser.add_argument("--checkpoint", type=str, default="",help="directory of the model checkpoint")
parser.add_argument("--o_checkpoint", type=str, default="",help="directory of the optimizer checkpoint")
parser.add_argument("--resume", action="store_true", help="directory of the teacher checkpoint for resuming")

parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument("--model_type", type=str, default="224", help="type of model, 224 means ViT-224, origin means DeiT-384", choices=["224", "origin"])

parser.add_argument("--ast_FT_dir", type=str, default=None, help="finetuning on AudioSet pretrained model, for VoxCeleb, ESC-50, SpeechCommands")
parser.add_argument("--resize_methods", type=str, default="PI",choices=["PI", "BL", "SC"],help="ways to resize, PI-resize, Bilinear, From scratch")
parser.add_argument("--wandb", action="store_true",help="use wandb or not")
parser.add_argument("--wandb_dir", type=str,help="wandb log directory")
parser.add_argument("--exp_name", type=str)

parser.add_argument('--only_time', help='if only flexify time domain', type=ast.literal_eval, default='False')
parser.add_argument('--only_freq', help='if only flexify freq domain', type=ast.literal_eval, default='False')
parser.add_argument('--patch_sizes', type=str, default="48,40,30,24,20,16,15,12,10,8", help="patch sizes for the flexiast model")
parser.add_argument('--t_model_path', type=str, default=None, help="model path for loading into the student")
parser.add_argument('--t_n_class', type=int, default=10, help="number of classes for the teacher model")
parser.add_argument('--t_fstride', type=int, default=10, help="soft split freq stride, overlap=patch_size-stride, of the model loaded into the student")
parser.add_argument('--t_tstride', type=int, default=10, help="soft split time stride, overlap=patch_size-stride, of the model loaded into the student")
parser.add_argument('--t_audio_length', type=int, default=1024, help="")
parser.add_argument('--t_imagenet_pretrain', type=ast.literal_eval, default='False')
# parser.add_argument('--t_audioset_pretrain', type=ast.literal_eval, default='False')
parser.add_argument("--t_model_type", type=str, default="224", help="type of model loaded into the student, 224 means ViT-224, origin means DeiT-384", choices=["224", "origin"])
parser.add_argument('--t_patch_size', type=int, default=8, help="the patch size of the model loaded")
parser.add_argument('--precompute_patch_embed', type=ast.literal_eval, default='False',help="the patch_embedding resizing matrix is precomputed for saving time,or not")
parser.add_argument('--skip_mlp', type=ast.literal_eval, default='False',help="To use the pretrained model esc50 and speechcommands, the mlp_head is not skiped")



import random
seed=1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
print('random seed: ', seed)

args = parser.parse_args()
args.patch_sizes = [int(x) for x in args.patch_sizes.split(",")]

if args.wandb:
    import wandb
    wandb.init(project="interspeech23", entity="kmmai", name=args.exp_name,dir=args.wandb_dir)

if args.model == 'ast':
    print('now train a audio spectrogram transformer model',args.model_type)

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 
                  'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean, 'std': args.dataset_std,'noise': args.noise,"skip_norm":args.skip_norm}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                      'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,"skip_norm":args.skip_norm}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  model_size=args.model_type,patch_size=args.patch_size)

    if args.ast_FT_dir is not None:
        #load a pretained model without mlp classifier
        print("====Start loading===")
        state_dict = torch.load(args.ast_FT_dir)
        out_dict = {}
        resized_weights = resize_weights(copy.deepcopy(state_dict),args.t_patch_size,args.patch_size,
                                            args.resize_methods,args.t_audio_length,args.audio_length) # should transform teacher's params for student to load
        print("Skip loading mlp_head",args.skip_mlp)
        for k, v in resized_weights.items(): # Adjust the name of dict
            # Why there is no mlp_head before?
            if args.skip_mlp and "mlp_head" in k:
                continue
            out_dict[k[7:]] = v
        audio_model.load_state_dict(out_dict,strict=False)
        print("====Finish loading===")

if args.model == 'flexiast':
    print('now train a flexiast model')

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 
                  'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean, 'std': args.dataset_std,'noise': args.noise,"skip_norm":args.skip_norm}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                      'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,"skip_norm":args.skip_norm}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.FlexiASTModel(
        label_dim=args.n_class, input_size=(128, args.audio_length), orig_patch_size=(args.patch_size, args.patch_size),
        orig_posemb_size=(128 // args.patch_size, args.audio_length // args.patch_size), patch_sizes=args.patch_sizes,
        # TODO: pos embed size, maybe change teh formula
        ast_pretrain=True, ast_params = dict(
            label_dim=args.t_n_class, fstride=args.t_fstride, tstride=args.t_tstride, input_fdim=128,
            input_tdim=args.t_audio_length, imagenet_pretrain=args.t_imagenet_pretrain,
            model_size=args.t_model_type, patch_size=args.t_patch_size
        ),
        model_path=args.t_model_path,load_backbone_only=False, precompute_patch_embed=args.precompute_patch_embed, verbose=False,
        resize_func=(vanilla_resample_patch_embed if args.resize_methods == 'BL' else resample_patch_embed),
        only_time=args.only_time, only_freq=args.only_freq,
    )

if args.resume:
    print("=========================")
    args.exp_dir = args.exp_dir + "_resume"

print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
# import ipdb;ipdb.set_trace()
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)



