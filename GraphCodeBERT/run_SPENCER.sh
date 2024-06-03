lang=python
code_model_dir=./models/dual_encoder/$lang
query_model_dir=./models/dual_encoder/$lang


CUDA_VISIBLE_DEVICES=0,1,2,3 python run_SPENCER.py \
    --code_model_dir $code_model_dir \
    --query_model_dir $query_model_dir \
    --tokenizer_name=microsoft/graphcodebert-base \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --test_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
