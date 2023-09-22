# measure the flexibility for VGGSound
CUDA_VISIBLE_DEVICES=2,3 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/measure_flexibility.py \
--dataset "vggsound" \
--num_class 309 \
--model_dir  "###DIRECTORY OF YOUR MODEL###" \
--epoch 15 \
--patch_size 16 \
--note "vggsound_supvs_init16" \
