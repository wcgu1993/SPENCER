lang=java
output_dir=./saved_models/cross_encoder/$lang
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,1 python3 run_cross_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --data_dir ../data/cross_encoder/$lang \
    --train_data_file ../data/cross_encoder/$lang/train.txt \
    --eval_data_file ../data/cross_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --code_length 150 \
    --nl_length 50 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1 | tee $output_dir/train.log