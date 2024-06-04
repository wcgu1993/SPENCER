lang=java
code_model_dir=./models/dual_encoder/$lang
query_model_dir=./models/dual_encoder/$lang
distllated_model_dir=./models/dual_encoder_12_to_3_layer/$lang
target_layer_num=3

CUDA_VISIBLE_DEVICES=1 python run_distillation.py \
    --language $lang \
    --code_model_dir $code_model_dir \
    --query_model_dir $query_model_dir \
    --distllated_model_dir $distllated_model_dir \
    --target_layer_num $target_layer_num \
    --tokenizer_name=microsoft/graphcodebert-base \
    --output_dir $output_dir \
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
