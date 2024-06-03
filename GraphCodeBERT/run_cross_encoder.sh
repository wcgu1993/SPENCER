lang=java #programming language
idx=1 #test batch idx
output_dir=./saved_models/cross_encoder/$lang


CUDA_VISIBLE_DEVICES=1 python3 run_cross_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/graphcodebert-base  \
    --data_dir ../data/cross_encoder/$lang \
    --do_test \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --test_data_file ../data/cross_encoder/test/$lang/batch_${idx}.txt \
    --test_result_dir ./new_test_results/$lang/${idx}_batch_result.txt