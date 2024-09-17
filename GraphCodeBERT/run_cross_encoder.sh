lang=python
mkdir -p ./saved_models/cross_encoder/$lang

CUDA_VISIBLE_DEVICES=0 python run_cross_encoder.py  \
    --output_dir=./saved_models/cross_encoder/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_test \
    --data_dir=../data/cross_encoder/$lang \
    --train_data_file=../data/cross_encoder/$lang/train.txt \
    --eval_data_file=../data/cross_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --input_length 384 \
    --test_result_dir ./results/$lang \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
    --seed 123456 2>&1| tee saved_models/cross_encoder/$lang/train.log