
model=flexiast
dataset=vggsound
set=full
imagenetpretrain=True

bal=full
lr=1e-5
epoch=20
tr_data=/home/jfeng/FJ/FlexiAST/egs/vggsound/data/datafiles/vgg_final_train.json
lrscheduler_start=2
lrscheduler_step=1
lrscheduler_decay=0.5


te_data=/home/jfeng/FJ/FlexiAST/egs/vggsound/data/datafiles/vgg_final_test.json
freqm=48
timem=192
mixup=0


dataset_mean=-5.0767093
dataset_std=4.4533687
audio_length=1024
noise=False

metrics=acc
loss=BCE
warmup=True


skip_norm=False
n_class=309
patch_size=16
t_patch_size=8
fstride=16
tstride=16
batch_size=12
label_csv="/home/jfeng/FJ/FlexiAST/egs/vggsound/data/class_labels_indices.csv"
resize_methods="PI"

# For the flexiast
patch_sizes="48,40,32,30,24,20,16,15,12,10,8"


t_model_path="### DIRECTORY OF PRETRAINED TEACHER ###"
t_n_class=309
t_fstride=8
t_tstride=8
t_audio_length=1024
t_imagenet_pretrain=True
t_audioset_pretrain=False
t_model_type="224"
precompute_patch_embed=True

exp_name=supervsFlex_T8_Spatch${patch_size}_Base224_-b$batch_size-m$mixup
exp_dir=/mnt/work2/users/fj/FlexiAST/supervs_flexiast/${dataset}/${exp_name}
# exp_dir=./exp/trival
# if [ -d $exp_dir ]; then
#   echo 'exp exist'
#   exit
# fi
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp_dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--patch_size ${patch_size} \
--t_patch_size ${t_patch_size} --patch_sizes ${patch_sizes} --t_model_path ${t_model_path} --t_n_class ${t_n_class} --t_fstride ${t_fstride} \
--t_tstride ${t_tstride} --t_audio_length ${t_audio_length} --t_imagenet_pretrain ${t_imagenet_pretrain} \
--t_model_type ${t_model_type} --t_patch_size ${t_patch_size} \
--precompute_patch_embed ${precompute_patch_embed} \
--wandb --exp_name ${exp_name} --wandb_dir ${exp_dir} \