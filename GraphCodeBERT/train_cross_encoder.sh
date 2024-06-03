lang=java
output_dir=./models/cross_encoder/$lang


CUDA_VISIBLE_DEVICES=0,1 python3 run_cross_encoder.py \
    --output_dir $output_dir \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --test_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee models/$lang/train.log