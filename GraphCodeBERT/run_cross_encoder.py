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

import argparse
import logging
import os
import pickle
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import multiprocessing
cpu_cont = 16


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 tokens,
                 ids,
                 idx, 
                 input_mask,
                 label,          

    ):
        self.tokens = tokens
        self.ids = ids
        self.idx=idx   
        self.input_mask=input_mask
        self.label=label   
        
        
def convert_examples_to_features(item):
    label,code,nl,tokenizer,args=item
    #code
    code_tokens=tokenizer.tokenize(code)
    nl_tokens=tokenizer.tokenize(nl)
    nl_tokens = nl_tokens[:args.input_length-3]
    code_tokens = code_tokens[:args.input_length-len(nl_tokens)-3]
    tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    #truncating
    ids =  tokenizer.convert_tokens_to_ids(tokens)
    idx = [i+tokenizer.pad_token_id + 1 for i in range(len(tokens))]
    padding_length=args.input_length-len(ids)
    idx+=[tokenizer.pad_token_id]*padding_length
    ids+=[tokenizer.pad_token_id]*padding_length
    input_mask = [1] * len(tokens) + [0] * padding_length    
 
    
    return InputFeatures(tokens,ids,idx,input_mask,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1].split('.')[0]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            with open(file_path) as f:
                for line in f:
                    line = line.strip().split('<CODESPLIT>')
                    label = int(line[0])  
                    nl = line[3]                  
                    code = line[4]
                    data.append((label,code,nl,tokenizer,args))
            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("tokens: {}".format([x.replace('\u0120','_') for x in example.tokens]))
                logger.info("ids: {}".format(' '.join(map(str, example.ids))))
                logger.info("idx: {}".format(example.idx))    
                logger.info("input mask: {}".format(' '.join([str(x) for x in example.input_mask])))           
                logger.info("label: {}".format(example.label))     
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
                 
        return (torch.tensor(self.examples[item].ids),
              torch.tensor(self.examples[item].idx), 
              torch.tensor(self.examples[item].input_mask), 
              torch.tensor(self.examples[item].label))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

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
    tr_num,tr_loss,best_acc=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            input_ids = batch[0].to(args.device)  
            position_idx = batch[1].to(args.device)
            input_mask = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            #get code and nl vectors
            outputs = model(input_ids=input_ids,position_ids=position_idx,attention_mask=input_mask,labels=labels)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['acc']>best_acc:
            best_acc=results['acc']
            logger.info("  "+"*"*20)  
            logger.info("  Best acc:%s",round(best_acc,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

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


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False, mode='dev', output=None):
    dataset = TextDataset(tokenizer, args, file_name, pool)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)
    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    preds = None
    out_label_ids = None
    for batch in dataloader:  
        with torch.no_grad():
            #get inputs
            input_ids = batch[0].to(args.device)  
            position_idx = batch[1].to(args.device)
            input_mask = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            #get code and nl vectors
            outputs = model(input_ids=input_ids,position_ids=position_idx,attention_mask=input_mask,labels=labels)
            _, logits = outputs[:2]
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)


    preds_label = np.argmax(preds, axis=1)
    result = compute_metrics(preds_label, out_label_ids)    
    if (mode == 'dev'):
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    elif (mode == 'test'):
        output_file = os.path.join(args.test_result_dir, output)
        if not os.path.exists(args.test_result_dir):
            os.makedirs(args.test_result_dir)
        with open(output_file, "w") as writer:
            logger.info("***** Output test results *****")
            all_logits = preds.tolist()
            for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                writer.write(str(logit[1]) + '\n')
            for key in sorted(result.keys()):
                print("%s = %s" % (key, str(result[key])))      
    

    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_result_dir", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--input_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
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
    
    pool = multiprocessing.Pool(cpu_cont)
    
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
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path)    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)

        test_data_dir = os.path.join(args.data_dir, 'test')
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                data_file = os.path.join(test_data_dir, file)
                result = evaluate(args, model, tokenizer, data_file, pool, mode='test', output=file)
                logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()