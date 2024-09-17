lang=python

CUDA_VISIBLE_DEVICES=0 python run_dual_encoder.py  \
    --output_dir=./saved_models/dual_encoder/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_test \
    --train_data_file=../data/dual_encoder/$lang/train.txt \
    --eval_data_file=../data/dual_encoder/$lang/valid.txt \
    --test_data_file=../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/dual_encoder/$lang/test.log