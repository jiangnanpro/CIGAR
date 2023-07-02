# CIGAR: Contrastive learning for GitHub Action Recommendation
RoBERTa based GitHub Actions recommendation model.

To train our model, run the run.py file with the following commands:

'''
python run.py \
    --output_dir=./saved_models/CIGAR_t50\
    --model_type=roberta \
    --tokenizer_name=roberta-base \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=../data/action_features_train.csv.gz  \
    --eval_data_file=../data/action_valid.csv.gz  \
    --test_data_file=../data/action_test.csv.gz  \
    --epoch 10 \
    --block_size 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_t50.log
'''
