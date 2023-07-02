# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import sys
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CosineSimilarity,LogSoftmax

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = 16
DATA_DIR = Path('../data')


from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def get_example(example):

    name_user, name_official, description_official, action_path, label, tokenizer, args = example
    
    name_user = tokenizer.tokenize(name_user)
    name_official = tokenizer.tokenize(name_official)
    description_official = tokenizer.tokenize(description_official)
    
        
    return convert_examples_to_features(name_user, name_official, description_official, action_path, label, tokenizer, args)


class InputFeatures(object):

    def __init__(self,
                 name_user_ids,
                 official_ids,
                 action_path,
    ):
        
        
        self.name_user_ids = name_user_ids
        self.official_ids = official_ids
        self.action_path = action_path


        
def phrase_tokenization(phrase, tokenizer, args):
    
    phrase_tokens = phrase[:args.block_size-2]
    phrase_tokens = [tokenizer.cls_token]+phrase_tokens+[tokenizer.sep_token]
    
    phrase_ids=tokenizer.convert_tokens_to_ids(phrase_tokens)
    padding_length = args.block_size - len(phrase_ids)
    phrase_ids+=[tokenizer.pad_token_id]*padding_length
    
    return phrase_ids
    
    
def convert_examples_to_features(name_user, name_official, description_official, action_path, label, tokenizer, args):
    
    name_user_ids = phrase_tokenization(name_user, tokenizer, args)
    name_official_ids = phrase_tokenization(name_official, tokenizer, args)
    description_official_ids = phrase_tokenization(description_official, tokenizer, args)
    
    official_ids = name_official_ids + description_official_ids
    
    return InputFeatures(name_user_ids,official_ids,action_path)

class TextDataset(Dataset):
    
    def __init__(self, tokenizer, args, file_path='train', evaluate=False, test=False, pool=None):
        
        
        postfix=file_path.split('/')[-1]
        self.examples = []
        index_filename=file_path
        
        
        if evaluate:
            
            logger.info("Creating features for validation from index file at %s ", index_filename)
            
            list_actions = (
                pd.read_csv(DATA_DIR / postfix, index_col = [0])
                .sort_values(by=['names_number'],ascending=False)
                .head(100)
                .to_dict('records')
            )
            
        else: 
            
            logger.info("Creating features for training from index file at %s ", index_filename)
            
            list_actions = (
                pd.read_csv(DATA_DIR / postfix, index_col = [0])
                .sort_values(by=['names_number'],ascending=False)
                .head(100)
                .to_dict('records')
            )
        
        
        list_names = [str(action['name_official']).lower() for action in list_actions]
        list_descriptions = [str(action['description_official']).lower() for action in list_actions]
        list_actions_path = [str(action['action']) for action in list_actions]
        list_names_users = [list(set(action['names_users'].lower().split(','))) for action in list_actions]
        list_names_users = [[name.strip() for name in names] for names in list_names_users]
        
        number_actions = len(list_names)
        
        data=[]
        
        for i in range(number_actions):
            for name_user in list_names_users[i]:
                data.append((name_user, list_names[i], list_descriptions[i], list_actions_path[i], 1, tokenizer, args))
            
        
        data = data[:]
        
        
        for i in tqdm(range(len(data))):
            self.examples.append(get_example(data[i]))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].name_user_ids),torch.tensor(self.examples[item].official_ids),self.examples[item].action_path


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,pool=None):
    dataset = TextDataset(tokenizer, args, file_path=args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file), evaluate=evaluate, test=test, pool=pool)
    return dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer,pool):
    """ Train the model """
    
    tau = 50

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_f1=0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        
        for step, batch in enumerate(bar):
            
            names_user = batch[0].to(args.device)
            official = batch[1].to(args.device)
            action_path = batch[2]
            
            model.train()
            user_vec, official_vec = model(names_user,official)
            
            # Contrastive loss function
            loss_temp = torch.zeros((len(user_vec),len(official_vec)*2-1),device=args.device, dtype=torch.float)
            for i in range(len(user_vec)):
                loss_temp[i][0] = (CosineSimilarity(dim=0)(user_vec[i],official_vec[i]) + 1) * 0.5 * tau
                indice = 1
                for j in range(len(user_vec)):
                    if i == j:
                        continue
                    temp = j
                    while action_path[i] == action_path[temp]:
                        temp = (temp + 1) % (len(user_vec))
                    loss_temp[i][indice] = (CosineSimilarity(dim=0)(user_vec[i],official_vec[temp]) + 1) * 0.5 * tau
                    indice += 1
                        
            con_loss = -LogSoftmax(dim=1)(loss_temp) # get the contrastive loss for each user's name in con_loss[x][0]
            con_loss = torch.sum(con_loss, dim=0)[0] # sum up with dim=0, the sum of contrastive loss for batch is con_loss[0]
            con_loss = con_loss / len(user_vec) # calculate the mean value as the contrastive loss for batch

            loss = con_loss
            

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)                 
                        # Save model checkpoint
                        
                        if results['eval_f1']>best_f1:
                            best_f1=results['eval_f1']
                            logger.info("  "+"*"*20)  
                            logger.info("  Best f1:%s",round(best_f1,4))
                            logger.info("  "+"*"*20)                          

                            checkpoint_prefix = 'checkpoint-best-f1'
                            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)                        
                            model_to_save = model.module if hasattr(model,'module') else model
                            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                            torch.save(model_to_save.state_dict(), output_dir)
                            logger.info("Saving model checkpoint to %s", output_dir)
                        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    '''
    checkpoint_prefix = 'checkpoint-final-model'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model,'module') else model
    output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
    torch.save(model_to_save.state_dict(), output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)
    '''
        
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="",pool=None,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True,pool=pool)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    avg_eval_loss = 0.0
    model.eval()
    logits=[]
    y_trues=[]
    
    
    cos_right = []
    cos_wrong = []
        
    for batch in tqdm(eval_dataloader):
        
        names_user = batch[0].to(args.device)
        official = batch[1].to(args.device)
        action_path = batch[2]
        
        
        with torch.no_grad():
            user_vec, official_vec = model(names_user,official)
        
        
        cos = CosineSimilarity(dim=1)(user_vec,official_vec)
        cos_right += cos.tolist()

        for i in range(len(user_vec)):
            nag_count = 0
            for j in range(len(user_vec)):
                if i == j:
                    continue
                if action_path[i] == action_path[j]:
                    continue
                cos_wrong += [CosineSimilarity(dim=0)(user_vec[i],official_vec[j]).item()]
                break
                nag_count += 1
                
                if nag_count == 6:
                    break
                    
    temp_best_f1 = 0
    temp_best_recall = 0
    temp_best_precision = 0
    temp_count = 0
    temp_error_count = 0
    temp_error_total = 0
    temp_total = 0
    temp_best_threshold = 0
    
    for i in tqdm(range(1, 100)):
        
        count = 0
        error_count = 0
        threshold = i/100
        for i in cos_right:
            if i >= threshold:
                count += 1
        total = len(cos_right)
        for i in cos_wrong:
            if i < threshold:
                error_count += 1
        error_total = len(cos_wrong)
        correct_recall = count/total
        if error_total-error_count+count == 0:
            continue
        precision = count/(error_total-error_count+count) 
        if precision+correct_recall == 0:
            continue
        F1 = 2*precision*correct_recall/(precision+correct_recall)
        if F1 > temp_best_f1:
            temp_best_f1 = F1
            temp_best_recall = correct_recall
            temp_best_precision = precision
            temp_count = count
            temp_error_count = error_count
            temp_error_total = error_total
            temp_total = total
            temp_best_threshold = threshold
            
    
    result = {
        "eval_recall": float(temp_best_recall),
        "eval_precision": float(temp_best_precision),
        "eval_f1": float(temp_best_f1),
        "eval_threshold":temp_best_threshold,
        
    }
    
    
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, prefix="",pool=None,best_threshold=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, test=True,pool=pool)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_preds=logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')
                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    

    ## Other parameters
    
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The input directory where the saved model checkpoints will be read.")
    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    
    parser.add_argument("--use_saved_model", action='store_true',
                        help="Whether to use the saved model or the original pretrained model CodeBERT.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        
                if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False,pool=pool)

        if args.local_rank == 0:
            torch.distributed.barrier()
        
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        
        if args.use_saved_model:
            logger.info("Now loading the saved model from %s", args.input_dir)
            input_dir = os.path.join(args.input_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(input_dir))

        global_step, tr_loss = train(args, train_dataset, model, tokenizer,pool)


    # Evaluation
    result = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix)) 
        #if args.use_saved_model:
        #    model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result=evaluate(args, model, tokenizer,pool=pool)
    
    
    # Testing
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        #if args.use_saved_model:
        #    model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer,pool=pool,best_threshold=result['eval_threshold'])

    return result


if __name__ == "__main__":
    main()

