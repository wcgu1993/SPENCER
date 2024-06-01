
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
from model import Model
# from cross_model import Model as Cross_model
from torch.nn import CrossEntropyLoss, MSELoss
from more_itertools import chunked
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, code, nl):
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

def convert_to_features(nl, codes, tokenizer, args):
    examples = []
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    for example in codes:
        code_tokens = tokenizer.tokenize(example)[:args.code_length-4]
        tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = args.code_length + args.nl_length - len(ids)
        ids += [tokenizer.pad_token_id]*padding_length
        examples.append(ids)
    return torch.tensor(examples)

# def convert_to_features(nl, code, tokenizer, args):
#     nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
#     code_tokens = tokenizer.tokenize(example)[:args.code_length-4]
#         tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
#         print(tokens)
#         ids = tokenizer.convert_tokens_to_ids(tokens)
#         padding_length = args.code_length + args.nl_length - len(ids)
#         ids += [tokenizer.pad_token_id]*padding_length
#         examples.append(ids)
#     return torch.tensor(examples)




def convert_examples_to_features(examples,tokenizer,args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(examples.code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl_tokens = tokenizer.tokenize(examples.nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        self.codes = []
        self.nls = []
    
        with open(file_path) as f:
           for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 2:
                    continue
                code = line[0]
                nl = line[1]
                self.codes.append(code)
                self.nls.append(nl)
                data.append(InputExample(code=code, nl=nl))
            
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

def ACC(real,predict):
    sum=0.0
    for val in real:
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1  
    return sum/float(len(real))

def evaluate(args, model, distilled_model, cross_model, tokenizer, file_name, language):

    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)
     
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(test_dataset))
    logger.info("  Num codes = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    distilled_model.eval()
    code_vecs = [] 
    nl_vecs = []
    n_processed = 0
    for batch in test_dataloader:
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs) 
            # nl_vec = model(nl_inputs=nl_inputs)
            nl_vec = distilled_model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 
            code_vecs.append(code_vec.cpu().numpy()) 
        n_processed += batch[0].size(0)
        # if n_processed > 10000:
        #     break
 
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    num_pool = n_processed // 1000
    # num_pool = 10
    accs1, accs3, accs5, mrrs = [], [], [], []
    softmax = torch.nn.Softmax(dim=1)
    # for i in tqdm(range(num_pool)):
    #     code_pool = code_vecs[i*1000:(i+1)*1000]
    #     codes = test_dataset.codes[i*1000:(i+1)*1000]
    #     for j in tqdm(range(1000)): # for i in range(pool_size):
    #         desc_vec = np.expand_dims(nl_vecs[i*1000+j], axis=0) # [1 x dim] 
    #         sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
    #         predict = sims.argsort()[::-1]
    #         predict1 = predict[:1]   
    #         predict1 = [int(k) for k in predict1]
    #         predict3 = predict[:3]   
    #         predict3 = [int(k) for k in predict3]
    #         predict5 = predict[:5]   
    #         predict5 = [int(k) for k in predict5]
    #         topk = predict[:5] 
    #         if j in topk:
    #             new_index = np.where(topk==j)  
    #             nl = test_dataset.nls[i*1000+j]
    #             selected_codes = [codes[k] for k in topk]
    #             # print(nl)
    #             # for k in topk:
    #             #     print(codes[k])
    #             inputs = convert_to_features(nl, selected_codes, tokenizer, args)
    #             inputs = inputs.to(args.device) 
    #             with torch.no_grad():
    #                 outputs = cross_model(inputs,attention_mask=inputs.ne(1)) 
    #             logits = outputs.logits
               
    #             # logits = softmax(logits)
    #             scores = logits[:,1]

    #             correct_score = logits[new_index[0],1]
    #             _, indices = torch.sort(scores, descending=True)
    #             _, indices = torch.sort(indices)
    #             rank = int(indices[new_index[0]]) + 1
    #             # print(output)
    #             # print(topk)
    #             # print(scores)
    #             # print("old: ", new_index[0], " new: ", rank)
    #             if rank <= 5:
    #                 accs5.append(1)
    #             else:
    #                 accs5.append(0)
                    
    #             if rank <= 3:
    #                 accs3.append(1)
    #             else: 
    #                 accs3.append(0)
                    
    #             if rank == 1:
    #                 accs1.append(1)
    #             else:
    #                 accs1.append(0)
                        
    #             mrrs.append(1/rank)
    #         else:    
    #             real = [j]
    #             accs1.append(ACC(real,predict1))
    #             accs3.append(ACC(real,predict3))
    #             accs5.append(ACC(real,predict5))
    #             index = np.where(predict==j)
    #             rank = int(index[0]) + 1
    #             mrrs.append(1/rank)
            # ndcgs.append(NDCG(real,predict))   
    file_dir = './results/{}'.format(language)
    for i in tqdm(range(num_pool)):
        code_pool = code_vecs[i*1000:(i+1)*1000]
        file = '{}_batch_result.txt'.format(i)
        with open(os.path.join(file_dir, file), encoding='utf-8') as f:
            batched_data = chunked(f.readlines(), 1000)
            
        for j, batch_data in enumerate(batched_data): # for i in range(pool_size):
        # for j in range(1000):
            try:
                desc_vec = np.expand_dims(nl_vecs[i*1000+j], axis=0) # [1 x dim] 
            except:
                print(len(nl_vecs))
                print(i)
                print(j)
            sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
            predict = sims.argsort()[::-1]
            predict1 = predict[:1]   
            predict1 = [int(k) for k in predict1]
            predict3 = predict[:3]   
            predict3 = [int(k) for k in predict3]
            predict5 = predict[:5]   
            predict5 = [int(k) for k in predict5]
            topk = predict[:2] 
            if j in topk:
            # if False:
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
    parser.add_argument("--language", default=None, type=str, required=True,
                        help="programming language.")
    parser.add_argument("--saved_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
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
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    config.num_hidden_layers = 3
    distilled_model = RobertaModel.from_pretrained(args.model_name_or_path, config = config)
 
    cross_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path) 
 


    
    model = Model(model)
    distilled_model = Model(distilled_model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    cross_model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
        cross_model = torch.nn.DataParallel(cross_model)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            saved_dir = os.path.join(args.saved_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(saved_dir))      
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            distilled_model_to_load = distilled_model.module if hasattr(distilled_model, 'module') else distilled_model  
            distilled_model_to_load.load_state_dict(torch.load(output_dir))    
            cross_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            cross_output_dir = os.path.join(args.cross_output_dir, '{}'.format(cross_checkpoint_prefix))  
            cross_model_to_load = cross_model.module if hasattr(cross_model, 'module') else cross_model  
            cross_model_to_load.load_state_dict(torch.load(cross_output_dir)) 
        model.to(args.device)
        distilled_model.to(args.device)
        cross_model.to(args.device)
        result = evaluate(args, model, distilled_model, cross_model, tokenizer,args.eval_data_file, args.language)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    main()