lang=python
CUDA_VISIBLE_DEVICES=0 python run_dual_encoder.py  \
    --output_dir ./saved_models/dual_encoder/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_eval \
    --do_test \
    --eval_data_file ../data/dual_encoder/$lang/valid.txt \
    --test_data_file ../data/dual_encoder/$lang/test.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456