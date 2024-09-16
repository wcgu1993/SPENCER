lang=java
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
output_dir=./models/dual_encoder/$lang


CUDA_VISIBLE_DEVICES=0 python run_dual_encoder.py \
    --output_dir $output_dir \
    --task_name codesearch \
    --model_type roberta \
	--evaluate_during_training \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
    --data_dir ../data/dual_encoder/$lang \
	--train_file train.txt \
	--dev_file valid.txt \
	--test_file test.txt \
    --num_train_epochs 10 \
	--max_seq_length 200 \
	--gradient_accumulation_steps 1 \
    --overwrite_output_dir \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 32 \
	--model_name_or_path $pretrained_model \
    --learning_rate 1e-5 \
	--logging_steps 5000 2>&1 | tee $output_dir/train.log