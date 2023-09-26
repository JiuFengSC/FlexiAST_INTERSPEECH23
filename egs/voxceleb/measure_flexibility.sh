CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/measure_time_flexibility.py \
--dataset "voxceleb" \
--num_class 1251 \
--model_dir  "/mnt/work2/users/fj/FlexiAST/time_only/voxceleb/reproduce_T8_Spatch16_Base224_-b12-m0/models/latest_audio_model15.pth" \
--epoch 15 \
--patch_size 16 \
--note "onlytime_t8"

# CUDA_VISIBLE_DEVICES=2 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/measure_time_flexibility.py \
# --dataset "voxceleb" \
# --num_class 1251 \
# --model_dir  "/mnt/work2/users/fj/FlexiAST/time_only/voxceleb/reproduce_T16_Spatch16_Base224_-b12-m0/models/latest_audio_model12.pth" \
# --epoch 12 \
# --patch_size 16 \
# --note "onlytime_t16"