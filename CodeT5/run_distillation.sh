lang=java
saved_dir=./saved_models/dual_encoder/$lang
output_dir=./saved_models/dual_encoder_12_to_3_layer_direct/$lang
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=1 python run_distillation.py \
    --language $lang \
    --saved_dir $saved_dir \
    --output_dir $output_dir \
    --model_name_or_path Salesforce/codet5-base  \
    --do_train \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1 | tee $output_dir/train.log