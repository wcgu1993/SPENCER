lang=java
output_dir=./models/dual_encoder/$lang
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,1 python run_dual_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1 | tee $output_dir/train.log