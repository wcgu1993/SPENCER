lang=python
mkdir -p ./saved_models/SPENCER/$lang

CUDA_VISIBLE_DEVICES=0 python run_SPENCER.py  \
    --output_dir=./saved_models/dual_encoder/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=../data/dual_encoder/$lang/train.txt \
    --eval_data_file=../data/dual_encoder/$lang/valid.txt \
    --test_data_file=../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --top_k 5 \
    --reduce_layer_num 3 \
    --nl_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/SPENCER/$lang/train.log