lang=java #programming language
idx=1 #test batch idx
output_dir=./models/cross_encoder/$lang

CUDA_VISIBLE_DEVICES=0 python run_cross_encoder.py \
    --output_dir $output_dir \
    --model_name_or_path microsoft/codebert-base \
    --task_name codesearch \
    --data_dir ../data/cross_encoder/$lang \
    --do_predict \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --num_train_epochs 10 \
    --max_seq_length 200 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --test_file batch_${idx}.txt \
    --pred_model_dir ./models/$lang/checkpoint-best/ 