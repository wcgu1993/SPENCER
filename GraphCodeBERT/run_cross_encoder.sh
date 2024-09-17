lang=java #programming language
output_dir=./models/cross_encoder/$lang

CUDA_VISIBLE_DEVICES=0 python run_cross_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/graphcodebert-base \
    --model_type roberta \
    --task_name codesearch \
    --data_dir ../data/cross_encoder/$lang \
    --do_predict \
    --train_file train.txt \
    --dev_file valid.txt \
    --num_train_epochs 10 \
    --max_seq_length 200 \
    --test_result_dir ./results/$lang \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --pred_model_dir ./models/cross_encoder/$lang/checkpoint-best/ 