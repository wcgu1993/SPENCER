# Training
lang=python
CUDA_VISIBLE_DEVICES=0 python run_cross_encoder.py  \
    --output_dir ./saved_models/cross_encoder/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file ../data/cross_encoder/$lang/train.txt \
    --eval_data_file ../data/cross_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --input_length 384 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
    --seed 123456 

