model=flexiast
dataset=voxceleb
set=full
imagenetpretrain=True

home_user=jfeng/FJ
exp_user=fj

bal=full
lr=1e-5
epoch=20
tr_data=/home/${home_user}/FlexiAST/egs/voxceleb/data/datafile/train_data.json
lrscheduler_start=2
lrscheduler_step=1
lrscheduler_decay=0.5
wa_start=1
wa_end=5

te_data=/home/${home_user}/FlexiAST/egs/voxceleb/data/datafile/test_data.json
freqm=48
timem=192
mixup=0


dataset_mean=-3.7614744
dataset_std=4.2011642
audio_length=1024
noise=False

metrics=acc
loss=CE
warmup=True
wa=True

n_class=1251
patch_size=16
fstride=16
tstride=16
batch_size=12
label_csv="/home/${home_user}/FlexiAST/egs/voxceleb/data/class_labels_indices.csv"
wandb_dir=/mnt/work2/users/${exp_user}/FlexiAST/flexiast
resize_methods="PI"
only_time=True

# For the flexiast
patch_sizes="48,40,32,30,24,20,16,12,10,8"

# t_* means the parameters of the ast model we will load into our own flexiast
t_model_path="/mnt/work2/users/mhamza/FlexiAST/voxceleb/exp/FT_Norm-CE-patch8_Base224_f8-t8-pTrue-b12/models/best_audio_model.pth"
t_n_class=1251
t_patch_size=8
t_fstride=8
t_tstride=8
t_audio_length=1024
t_imagenet_pretrain=True
t_model_type="224"
precompute_patch_embed=True

exp_dir=/mnt/work2/users/${exp_user}/FlexiAST/time_only/${dataset}/reproduce_T8_Spatch${patch_size}_Base224_-b$batch_size-m$mixup
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp_dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
--patch_size ${patch_size} \
--t_patch_size ${t_patch_size} \
--exp_name supervsFlex_${dataset}_${resize_methods}_T8_Spatch${patch_size}_-b$batch_size --wandb_dir ${wandb_dir} \
--patch_sizes ${patch_sizes} --t_model_path ${t_model_path} --t_n_class ${t_n_class} --t_fstride ${t_fstride} \
--t_tstride ${t_tstride} --t_audio_length ${t_audio_length} --t_imagenet_pretrain ${t_imagenet_pretrain} \
--t_model_type ${t_model_type} --t_patch_size ${t_patch_size} \
--precompute_patch_embed ${precompute_patch_embed} \
--only_time ${only_time} \
