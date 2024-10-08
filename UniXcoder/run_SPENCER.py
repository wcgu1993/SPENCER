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
import torch.nn.functional as F
import random
import torch
import numpy as np
from tqdm import tqdm, trange
from model import Model
from more_itertools import chunked
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

        
def convert_examples_to_features(example,tokenizer,args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(example['code'])[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl_tokens = tokenizer.tokenize(example['nl'])[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip().split('<CODESPLIT>')
                code = line[0]
                nl = line[1]    
                temp = {}      
                temp['code'] = code
                temp['nl'] = nl    
                data.append(temp)

        for example in data:
            self.examples.append(convert_examples_to_features(example,tokenizer,args))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, teacher_model, distilled_model, tokenizer, saved_dir):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(distilled_model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.eval()
    teacher_model.eval()
    # model.resize_token_embeddings(len(tokenizer))
    distilled_model.zero_grad()
    
    distilled_model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = teacher_model(nl_inputs=nl_inputs)
            
            distilled_nl_vec = distilled_model(nl_inputs=nl_inputs)

            code_vec = F.normalize(code_vec, dim = 1)
            nl_vec = F.normalize(nl_vec, dim = 1)
            distilled_nl_vec = F.normalize(distilled_nl_vec, dim = 1)
            
            dual_similarity = torch.einsum("ab,cb->ac",code_vec,nl_vec)

            distilled_dual_similarity = torch.einsum("ab,cb->ac", code_vec, distilled_nl_vec)
            distilled_query_similarity = torch.sum(torch.mul(nl_vec, distilled_nl_vec), dim=1)
            query_similarity = torch.ones_like(distilled_query_similarity)
            
            loss = torch.abs(dual_similarity - distilled_dual_similarity).sum() + torch.abs(query_similarity - distilled_query_similarity).sum()

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
        results = evaluate(args, model, tokenizer,args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(saved_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

def ACC(real,predict):
    sum=0.0
    for val in real:
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1  
    return sum/float(len(real))

def evaluate(args, model, distilled_model, tokenizer):
    dataset = TextDataset(tokenizer, args, args.test_data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)
    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    n_processed = 0
    for batch in dataloader: 
        code_inputs = batch[0].to(args.device)  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = distilled_model(nl_inputs=nl_inputs) 
            code_vec = model(code_inputs=code_inputs)
            nl_vecs.append(nl_vec.cpu().numpy()) 
            code_vecs.append(code_vec.cpu().numpy())        
            n_processed += batch[0].size(0)
    
    model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    num_pool = n_processed // 1000
    accs1, accs3, accs5, mrrs = [], [], [], []
    
    file_dir = './results/{}'.format(args.lang)

    for i in tqdm(range(num_pool)):
        code_pool = code_vecs[i*1000:(i+1)*1000]
        file = 'batch_{}.txt'.format(i)
        with open(os.path.join(file_dir, file), encoding='utf-8') as f:
            batched_data = chunked(f.readlines(), 1000)
            
        for j, batch_data in enumerate(batched_data): # for i in range(pool_size):
            desc_vec = np.expand_dims(nl_vecs[i*1000+j], axis=0) # [1 x dim] 
            sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
            predict = sims.argsort()[::-1]
            predict1 = predict[:1]   
            predict1 = [int(k) for k in predict1]
            predict3 = predict[:3]   
            predict3 = [int(k) for k in predict3]
            predict5 = predict[:5]   
            predict5 = [int(k) for k in predict5]
            topk = predict[:args.top_k] 
            if j in topk:
                correct_score = float(batch_data[j].strip().split('<CODESPLIT>')[-1])
                scores = np.array([float(batch_data[k].strip().split('<CODESPLIT>')[-1]) for k in topk])
                rank = np.sum(scores >= correct_score)
                if rank <= 5:
                    accs5.append(1)
                else:
                    accs5.append(0)
                    
                if rank <= 3:
                    accs3.append(1)
                else: 
                    accs3.append(0)
                    
                if rank == 1:
                    accs1.append(1)
                else:
                    accs1.append(0)
                        
                mrrs.append(1/rank)     
            else:    
                real = [j]
                accs1.append(ACC(real,predict1))
                accs3.append(ACC(real,predict3))
                accs5.append(ACC(real,predict5))
                index = np.where(predict==j)
                rank = int(index[0]) + 1
                mrrs.append(1/rank)
        
    result = {
        'eval_acc1': float(np.mean(accs1)),
        'eval_acc3': float(np.mean(accs3)), 
        'eval_acc5': float(np.mean(accs5)),
        "eval_mrr":float(np.mean(mrrs))
    }

    return result
                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--lang", default=None, type=str,
                        help="language.") 
        
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
    parser.add_argument("--top_k", default=None, type=int, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--reduce_layer_num", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    
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
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:
        last_layer_num = 12
        teacher_layer_num = 12
        ta_layer_num = teacher_layer_num - args.reduce_layer_num
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        config.num_hidden_layers = max(last_layer_num - args.reduce_layer_num, 1)
        distilled_model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)   
        distilled_model = Model(distilled_model)
        distilled_model.load_state_dict(torch.load(output_dir),strict=False)   
        distilled_model.to(args.device) 
        logger.info("Training/evaluation parameters %s", args)
    
        
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)  
            distilled_model = torch.nn.DataParallel(distilled_model)

   
        saved_dir = os.path.join(args.output_dir, "model_{}_to_{}_layer".format(last_layer_num, config.num_hidden_layers))
        ta_model = train(args, model, model, distilled_model, tokenizer, saved_dir)
        teacher_model = model
        results = evaluate(args, model, distilled_model, tokenizer)
        original_mrr = results['mrr']
        distilled_results = evaluate(args, model, distilled_model, tokenizer)
        distilled_mrr = distilled_results['mrr']
        best_distillation_dir = args.output_dir

        if distilled_mrr / original_mrr < 0.01:
            best_distillation_dir = saved_dir

        last_layer_num = config.num_hidden_layers

        while config.num_hidden_layers > 1 and distilled_mrr / original_mrr < 0.01:
            config.num_hidden_layers = max(last_layer_num - args.reduce_layer_num, 1)
            distilled_model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)  
            distilled_model = Model(distilled_model)
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin' 
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            distilled_model.load_state_dict(torch.load(output_dir),strict=False)   
            distilled_model.to(args.device)

            teacher_saved_dir = os.path.join(args.output_dir, "model_{}_to_{}_layer".format(teacher_layer_num, config.num_hidden_layers))
            teacher_distilled_model = train(args, model, teacher_model, distilled_model, tokenizer, teacher_saved_dir)

            ta_saved_dir = os.path.join(args.output_dir, "model_{}_to_{}_layer".format(ta_layer_num, config.num_hidden_layers))
            ta_distilled_model = train(args, model, ta_model, distilled_model, tokenizer, ta_saved_dir)
            teacher_results = evaluate(args, model, teacher_distilled_model, tokenizer)
            teacher_mrr = teacher_results['mrrs']
            ta_results = evaluate(args, model, ta_distilled_model, tokenizer)
            ta_mrr = ta_results['mrr']
            if teacher_mrr > ta_mrr:
                if teacher_mrr / original_mrr:
                    best_distillation_dir = teacher_saved_dir
                ta_model = teacher_distilled_model
                ta_layer_num = config.num_hidden_layers
                distilled_mrr = teacher_mrr
            else:
                if ta_mrr / original_mrr:
                    best_distillation_dir = ta_saved_dir
                teacher_model = ta_model
                teacher_layer_num = ta_layer_num
                ta_model = ta_distilled_model
                ta_layer_num = config.num_hidden_layers
                distilled_mrr = ta_mrr

            last_layer_num = config.num_hidden_layers
        
        with open(os.path.join(args.output_dir, 'best-distillation.txt'), 'w') as f:
            f.write(best_distillation_dir)

    if args.do_eval:
        with open(os.path.join(args.output_dir, 'best-distillation.txt'), 'r') as f:
            query_model_dir = f.readline().strip()
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        try:
            config.num_hidden_layers = int(query_model_dir.split('_')[-2])
        except:
            config.num_hidden_layers = 12
        query_model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)   
        query_model = Model(query_model)
        output_dir = os.path.join(query_model_dir, '{}'.format(checkpoint_prefix))  
        query_model.load_state_dict(torch.load(output_dir),strict=False)   
        model.to(args.device)
        query_model.to(args.device)        
        result = evaluate(args, model, query_model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))



if __name__ == "__main__":
    main()