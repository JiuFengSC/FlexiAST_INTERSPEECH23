import torch
import argparse
import os
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
from traintest import train, validate
import torch.nn as nn

print (os.getcwd())

import logging
import math
from typing import List, Tuple, Optional, Union

import torch.nn.functional as F

import sys
from utilities import *
import time
import pickle
from tqdm import tqdm
from models.flexi_ast import resample_patch_embed

from timm.layers.helpers import to_2tuple
import copy
import argparse


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_dir", type=str, required=True,help="best model directory")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_class", type=int, required=True,help="number of class")
parser.add_argument("--epoch", type=int, default=1,help="number of class")
parser.add_argument("--patch_size", type=int, default=8,help="number of class")
parser.add_argument("--t_patch_size", type=int, default=8,help="number of class")
parser.add_argument("--note", type=str, default="",help="number of class")
parser.add_argument("--audio_length", type=int, default=1024,help="length of spectrogram")
parser.add_argument("--model_type", type=str, default="224", help="type of model", choices=["224", "origin"])





args = parser.parse_args()

print(f"{args.epoch}+{args.model_dir}")

def validate(audio_model, val_loader, loss_fn, patch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        with tqdm(val_loader) as pbar:
            for i, (audio_input, labels, path) in enumerate(pbar):
                # print(audio_input.device)
                audio_input = audio_input.to(device)

                # compute output
                audio_output = audio_model(audio_input,patch)
                audio_output = torch.sigmoid(audio_output)
                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                # compute the loss
                labels = labels.to(device)
                loss = loss_fn(audio_output, labels)
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

    return stats, loss




audio_model = models.FlexiASTModel(
    label_dim=args.num_class, input_size=(128, args.audio_length), orig_patch_size=(16,16),
    orig_posemb_size=(128 // 16, args.audio_length // 16), patch_sizes=16,
    ast_pretrain=True, ast_params = dict(
        label_dim=args.num_class, fstride=args.t_patch_size, tstride=args.t_patch_size, input_fdim=128,
        input_tdim=args.audio_length, imagenet_pretrain=True,
        audioset_pretrain=False, model_size=args.model_type, patch_size=args.t_patch_size
    ),
    model_path=None,load_backbone_only=False, precompute_patch_embed=False, verbose=False
)
state_dict = torch.load(args.model_dir)
out_dict = {}
for k, v in state_dict.items(): # Adjust the name of dict
    # if "v.patch_embed" in k or "v.pos_embed" in k: # we replace this two components with new classes, but keep the structure the same.
    #     continue
    out_dict[k[7:]] = v
# import ipdb; ipdb.set_trace()
# RuntimeError: Error(s) in loading state_dict for FlexiASTModel:
#         size mismatch for v.pos_embed: copying a param with shape torch.Size([1, 2049, 768]) from checkpoint, the shape in current model is torch.Size([1, 513, 768]).
#         size mismatch for v.patch_embed.proj.weight: copying a param with shape torch.Size([768, 1, 8, 8]) from checkpoint, the shape in current model is torch.Size([768, 1, 16, 16]).
audio_model.load_state_dict(out_dict)


audio_model.v.patch_embed = None
audio_model.v.pos_embed = None

new_patch_list = [8,10,12,16,20,24,30,32,40,48]


mAP_list = []
acc_list = []
AUC_list = []
# img size = 1024*128
for patch in new_patch_list:

    audio_model.eval()
    batch_size = 12
    if patch == 4:
        batch_size = 6
    elif patch == 6:
        batch_size = 8

    if args.dataset == "audioset":
        val_conf = {'num_mel_bins': 128, 'target_length': 1024,
                'freqm': 0, 'timem': 0, 'mixup': 0,
                'dataset': "audioset", 'mode': 'evaluation', 'mean': -4.2677393,
                'std': 4.5689974, 'noise': False, "skip_norm": False}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset("../audioset/data/datafiles/eval.json",
            label_csv="../audioset/data/class_labels_indices.csv",
            audio_conf=val_conf),
            batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True)
    elif args.dataset == "vggsound":
        val_conf = {'num_mel_bins': 128, 'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': "vggsound", 'mode': 'evaluation', 'mean': -5.0767093,
        'std': 4.4533687, 'noise': False, "skip_norm": False}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset("/home/jfeng/FJ/FlexiAST/egs/vggsound/data/datafiles/vgg_final_test.json",
            label_csv="../vggsound/data/class_labels_indices.csv",
            audio_conf=val_conf),
            batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True)
    elif args.dataset == "voxceleb":
        val_conf = {'num_mel_bins': 128, 'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': "voxceleb", 'mode': 'evaluation', 'mean': -3.7614744,
        'std': 4.2011642, 'noise': False, "skip_norm": False}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset("../voxceleb/data/datafile/test_data.json",
            label_csv="../voxceleb/data/class_labels_indices.csv",
            audio_conf=val_conf),
            batch_size=batch_size, shuffle=False,
            num_workers=32, pin_memory=True)
    elif args.dataset == "esc50":
        val_conf = {'num_mel_bins': 128, 'target_length': 512,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': "esc50", 'mode': 'evaluation', 'mean': -6.6268077,
        'std': 5.358466, 'noise': False, "skip_norm": False}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset("/home/jfeng/FJ/FlexiAST/egs/esc50/data/datafiles/esc_eval_data_1.json",
            label_csv="/home/jfeng/FJ/FlexiAST/egs/esc50/data/esc_class_labels_indices.csv",
            audio_conf=val_conf),
            batch_size=48, shuffle=False,
            num_workers=32, pin_memory=True)
    elif args.dataset == "speechcommands":
        val_conf = {'num_mel_bins': 128, 'target_length': 128,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'dataset': "esc50", 'mode': 'evaluation', 'mean': -6.845978,
        'std': 5.5654526, 'noise': False, "skip_norm": False}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset("/home/mhamza/FlexiAST/egs/speechcommands/data/datafiles/speechcommand_eval_data.json",
            label_csv="/home/mhamza/FlexiAST/egs/speechcommands/data/speechcommands_class_labels_indices.csv",
            audio_conf=val_conf),
            batch_size=128, shuffle=False,
            num_workers=32, pin_memory=True)

    loss_fn = nn.BCEWithLogitsLoss()

    stats,loss = validate(audio_model, val_loader, loss_fn, patch)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']
    mAP_list.append(mAP)
    acc_list.append(acc)
    # Get the middel value of each precisions
    middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
    middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
    average_precision = np.mean(middle_ps)
    average_recall = np.mean(middle_rs)
    print(f"===========Patch {patch}============")
    print("mAP: {:.6f}".format(mAP))
    print("acc: {:.6f}".format(acc))
    print("AUC: {:.6f}".format(mAUC))
    print("Avg Precision: {:.6f}".format(average_precision))
    print("Avg Recall: {:.6f}".format(average_recall))
    print("d_prime: {:.6f}".format(d_prime(mAUC)))
    # del audio_model

print(mAP_list)
print(acc_list)

with open(f'{args.note}flex_{args.dataset}_{args.epoch}.csv', 'w') as f:
    for i in range(len(new_patch_list)):
        string1 = str(new_patch_list[i])+", "+str(mAP_list[i])+", "+str(acc_list[i])+"\n"
        f.write(string1)
