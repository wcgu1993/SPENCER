lang=python
saved_dir=./saved_models/new_dual_encoder_8/$lang
output_dir=./saved_models/revised_dual_encoder_12_to_3_rerun/$lang
cross_output_dir=./saved_models/cross_encoder/$lang
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,1,2,3 python run_SPENCER.py \
    --language $lang \
    --saved_dir $saved_dir \
    --output_dir $output_dir \
    --cross_output_dir $cross_output_dir \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_eval \
    --train_data_file ../data/dual_encoder/$lang/train.txt \
    --eval_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 64 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456