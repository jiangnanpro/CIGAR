# CIGAR: Contrastive learning for GHA Recommendation

This repository contains the code and the dataset of our paper.

- **CIGAR** contains the source code of our model
- **CIGAR_variants** contains the variants of our model (i.e. RoBERTa w/ or w/o finetune)
- **data** folder contains our dataset, along with the notebook for building the dataset.
- **plots** folder contains plots for data visualization.


To train and save our model, run the run.py file with the following commands:

```
python run.py \
    --output_dir=./saved_models/CIGAR_t50 \
    --model_type=roberta \
    --tokenizer_name=roberta-base \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=../../data/train.csv.gz  \
    --eval_data_file=../../data/valid.csv.gz  \
    --test_data_file=../../data/test.csv.gz  \
    --epoch 10 \
    --block_size 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_t50.log
```

- **results** folder contains the predictions provided by our model with various $\tau$ selected. The structure of the results is as follows:
  
    | name        | actual_action | suggested_actions |
    | ----------- | ----------- | ------------------ |
    | upload source artifact     | 0      | \[0, 22, 13, 51, 88\] |
    | ... | ... | ... |
    | select latest xcode   | 99        | \[99, 36, 37, 76, 14\] |
