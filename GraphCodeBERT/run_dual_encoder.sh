lang=python
output_dir=./saved_models/revised_dual_encoder_12_to_3_rerun/$lang
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,1,2,3 python run_dual_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/graphcodebert-base  \
    --do_eval \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456