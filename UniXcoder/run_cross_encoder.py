
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
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, code, nl, label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.code = code
        self.nl = nl
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 tokens,
                 ids,
                 label

    ):
        self.tokens = tokens
        self.ids = ids
        self.label = int(label)



def convert_examples_to_features(examples,tokenizer,args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(examples.code)[:args.code_length-4]
    nl_tokens = tokenizer.tokenize(examples.nl)[:args.nl_length-4]
    # tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.code_length + args.nl_length - len(ids)
    ids += [tokenizer.pad_token_id]*padding_length
    
    return InputFeatures(tokens,ids,examples.label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        load_cache = False
        if "train" in file_path:
            ttype = "train"
            load_cache = True
        elif "valid" in file_path:
            ttype = "valid"
            load_cache = True
        process_num = 0

        if load_cache:
            cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}'.format(
                ttype,
                list(filter(None, args.model_name_or_path.split('/'))).pop()))

        # if os.path.exists(cached_features_file):
        try:
            logger.info("Loading features from cached file %s", cached_features_file)
            self.examples = torch.load(cached_features_file)
        except:
            logger.info("Creating features from dataset file at %s", args.data_dir)
    
            with open(file_path) as f:
                for line in f.readlines():
                    line = line.strip().split('<CODESPLIT>')
                    if len(line) != 5:
                        continue
                    label = line[0]
                    nl= line[3]
                    code = line[4]
                    data.append(InputExample(code=code, nl=nl, label=label))

            
            for example in data:
                self.examples.append(convert_examples_to_features(example,tokenizer,args))
                process_num += 1
                if process_num % 100000 == 0:
                    logger.info("Loading data number %s", str(process_num))

            if load_cache:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.examples, cached_features_file)
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("tokens: {}".format([x.replace('\u0120','_') for x in example.tokens]))
                logger.info("ids: {}".format(' '.join(map(str, example.ids))))                         
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].ids),torch.tensor(self.examples[i].label))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ACC(real,predict):
    sum=0.0
    for val in real:
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1  
    return sum/float(len(real))

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)    
            label = batch[1].to(args.device)
            outputs = model(inputs,attention_mask=inputs.ne(1),labels=label) 
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['acc']>best_acc:
            best_acc = results['acc']
            logger.info("  "+"*"*20)  
            logger.info("  Best acc:%s",round(best_acc,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            model_to_save.save_pretrained(output_dir) 
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,eval_when_training=False):

    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)
     
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(test_dataset))
    logger.info("  Num codes = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    preds = []
    labels = []
    for batch in test_dataloader:
        inputs = batch[0].to(args.device)    
        label = batch[1].to(args.device)
        with torch.no_grad():
            outputs = model(inputs,attention_mask=inputs.ne(1),labels=label) 
            tmp_eval_loss, logits = outputs[:2]
            preds.append(logits.cpu().numpy()) 
            labels.append(label.cpu().numpy()) 

    model.train()    
    preds = np.concatenate(preds,0)
    labels = np.concatenate(labels,0)
    preds_label = np.argmax(preds, axis=1)
    result = compute_metrics(preds_label, labels)

    return result

def predict(args, model, tokenizer,file_name):

    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)
     
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(test_dataset))
    logger.info("  Num codes = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    outputs = []
    codes = []
    nls = []
    softmax = torch.nn.Softmax(dim=1)
    for batch in tqdm(test_dataloader):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            output = model(inputs,attention_mask=inputs.ne(1),labels=label) 
            tmp_eval_loss, logits = output[:2]
            # output = softmax(output)
            outputs.append(logits.cpu().numpy()) 

    outputs = np.concatenate(outputs,0)
    answers = outputs[:,1]
    output_test_file = args.test_result_dir
    output_dir = os.path.dirname(output_test_file)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with open(output_test_file, "w") as writer:
        logger.info("***** Output test results *****")
        for logit in answers:
            writer.write(str(logit) + '\n')
                     
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--test_result_dir", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path) 
 
    
    # model = Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:
        train(args, model, tokenizer)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model = RobertaForSequenceClassification.from_pretrained(output_dir)    
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model = RobertaForSequenceClassification.from_pretrained(output_dir)       
        model.to(args.device)
        result = predict(args, model, tokenizer, args.test_data_file)


if __name__ == "__main__":
    main()