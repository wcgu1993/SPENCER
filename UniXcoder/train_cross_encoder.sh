lang=java
pretrained_model=microsoft/unixcoder-base  #Roberta: roberta-base
output_dir=./models/cross_encoder/$lang

CUDA_VISIBLE_DEVICES=0 python run_cross_encoder.py \
    --model_type roberta \
    --task_name codesearch \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --train_file train.txt \
    --dev_file valid.txt \
    --num_train_epochs 10 \
    --max_seq_length 200 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --data_dir ../data/cross_encoder/$lang \
    --output_dir $output_dir  \
    --model_name_or_path $pretrained_model \
    --learning_rate 1e-5 \
    --seed 123456 2>&1 | tee $output_dir/train.log


