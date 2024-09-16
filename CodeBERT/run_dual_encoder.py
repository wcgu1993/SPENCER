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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
import argparse
import glob
import logging
import os
import random
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaModel,
                          RobertaTokenizer)

from dual_utils import (convert_examples_to_features, processors)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, code_model, query_model, tokenizer, code_optimizer, query_optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    code_scheduler = get_linear_schedule_with_warmup(code_optimizer, args.warmup_steps, t_total)
    query_scheduler = get_linear_schedule_with_warmup(query_optimizer, args.warmup_steps, t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    code_model.zero_grad()
    query_model.zero_grad()
    global_step = 0
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    code_model.train()
    query_model.train()
    for idx in range(args.num_train_epochs):
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            code_inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            summary_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
            code_outputs = code_model(**code_inputs)
            summary_outputs = query_model(**summary_inputs)
            
            aug_code_outputs = code_model(**code_inputs)
            aug_summary_outputs = query_model(**summary_inputs)

            code_outputs = code_outputs.pooler_output
            summary_outputs = summary_outputs.pooler_output
            aug_code_outputs = aug_code_outputs.pooler_output
            aug_summary_outputs = aug_summary_outputs.pooler_output

            code_outputs = F.normalize(code_outputs, dim = 1)
            summary_outputs = F.normalize(summary_outputs, dim = 1)
            aug_code_outputs = F.normalize(aug_code_outputs, dim = 1)
            aug_summary_outputs = F.normalize(aug_summary_outputs, dim = 1)

            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            
            scores = torch.einsum("ab,cb->ac",code_outputs,summary_outputs)
            contra_loss = loss_fct(scores*20, torch.arange(code_outputs.size(0), device=scores.device))

            scores = torch.einsum("ab,cb->ac",code_outputs,aug_code_outputs)
            code_contra_loss = loss_fct(scores*20, torch.arange(code_outputs.size(0), device=scores.device))

            scores = torch.einsum("ab,cb->ac",summary_outputs,aug_summary_outputs)
            summary_contra_loss = loss_fct(scores*20, torch.arange(code_outputs.size(0), device=scores.device))
            
            loss = contra_loss.sum() + code_contra_loss.sum()+ summary_contra_loss.sum()  

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(code_model.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(query_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                code_optimizer.step()
                query_optimizer.step()
                code_scheduler.step()  # Update learning rate schedule
                query_scheduler.step()  # Update learning rate schedule
                code_model.zero_grad()
                query_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, code_model, query_model, tokenizer, 'dev')
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info('loss %s', str(tr_loss - logging_loss))
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, code_model, query_model, tokenizer, 'dev')

            if (results['acc'] > best_acc):
                best_acc = results['acc']
                code_output_dir = os.path.join(args.output_dir, 'code_model')
                query_output_dir = os.path.join(args.output_dir, 'query_model')
                code_model.save_pretrained(code_output_dir)
                query_model.save_pretrained(query_output_dir)
                torch.save(args, os.path.join(args.output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", args.output_dir)

                torch.save(code_optimizer.state_dict(), os.path.join(code_output_dir, "code_optimizer.pt"))
                torch.save(query_optimizer.state_dict(), os.path.join(query_output_dir, "query_optimizer.pt"))
                torch.save(code_scheduler.state_dict(), os.path.join(code_output_dir, "code_scheduler.pt"))
                torch.save(query_scheduler.state_dict(), os.path.join(query_output_dir, "query_scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", args.output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def ACC(real,predict):
    sum=0.0
    for val in real:
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1  
    return sum/float(len(real))

def MAP(real,predict):
    sum=0.0
    for id, val in enumerate(real):
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+(id+1)/float(index+1)
    return sum/float(len(real))

def MRR(real, predict):
    sum=0.0
    for val in real:
        try: index = predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1.0/float(index+1)
    return sum/float(len(real))

def NDCG(real, predict):
    dcg=0.0
    idcg=IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i+1
            dcg +=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
    return dcg/float(idcg)

def IDCG(n):
    idcg=0
    itemRelevance=1
    for i in range(n): idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
    return idcg


def evaluate(args, code_model, query_model, tokenizer, ttype):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype=ttype)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        accs1, accs3, accs5 = [],[],[]
        nl_vecs, code_vecs = [], []
        n_processed = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            code_model.eval()
            query_model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                code_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
                summary_inputs = {'input_ids': batch[2],
                            'attention_mask': batch[3]}
                code_outputs = code_model(**code_inputs).pooler_output
                summary_outputs = query_model(**summary_inputs).pooler_output
                code_vec = F.normalize(code_outputs, dim = 1)
                nl_vec = F.normalize(summary_outputs, dim = 1)
                nl_vecs.append(nl_vec.cpu().numpy()) 
                code_vecs.append(code_vec.cpu().numpy()) 
                n_processed += batch[0].size(0)

        code_model.train()  
        query_model.train()    
        code_vecs = np.concatenate(code_vecs,0)
        nl_vecs = np.concatenate(nl_vecs,0)
        num_pool = n_processed // 1000
        accs1, accs3, accs5, mrrs = [], [], [], []
        for i in range(num_pool):
            code_pool = code_vecs[i*1000:(i+1)*1000]
            for j in range(1000): # for i in range(pool_size):
                desc_vec = np.expand_dims(nl_vecs[i*1000+j], axis=0) # [1 x dim] 
                sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
                predict = sims.argsort()[::-1]
                predict1 = predict[:1]   
                predict1 = [int(k) for k in predict1]
                predict3 = predict[:3]   
                predict3 = [int(k) for k in predict3]
                predict5 = predict[:5]   
                predict5 = [int(k) for k in predict5]
                real = [j]
                accs1.append(ACC(real,predict1))
                accs3.append(ACC(real,predict3))
                accs5.append(ACC(real,predict5))
                index = np.where(predict==j)
                rank = index[0] + 1
                mrrs.append(1/rank)
            # ndcgs.append(NDCG(real,predict))                     
        
        result = {
            'eval_acc1': float(np.mean(accs1)),
            'eval_acc3': float(np.mean(accs3)), 
            'eval_acc5': float(np.mean(accs5)),
            "eval_mrr":float(np.mean(mrrs))
        }               
        
        
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results {} *****")
            logger.info('acc1: %f, acc3: %f, acc5: %f, mrr: %f', np.mean(accs1), np.mean(accs3), np.mean(accs5), np.mean(mrrs))
            writer.write('acc1: %f\n' % np.mean(accs1))

    return {'acc1':np.mean(accs1), 'acc3': np.mean(accs3), 'acc5': np.mean(accs5),'mrr': np.mean(mrrs)}  

def load_and_cache_examples(args, task, tokenizer, ttype='train'):
    processor = processors[task]()
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    except:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples = processor.get_test_examples(args.data_dir, args.test_file)
        

        features = convert_examples_to_features(examples, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_code_input = torch.tensor([f.code_input for f in features], dtype=torch.long)
    all_code_mask = torch.tensor([f.code_mask for f in features], dtype=torch.long)
    all_summary_input = torch.tensor([f.summary_input for f in features], dtype=torch.long)
    all_summary_mask = torch.tensor([f.summary_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_code_input, all_code_mask, all_summary_input, all_summary_mask)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='codesearch', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
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
    parser.add_argument("--num_train_epochs", default=3.0, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="test file")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    args = parser.parse_args()



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

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        finetuning_task=args.task_name)
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    code_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    query_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    code_model.to(args.device)
    query_model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    code_optimizer_grouped_parameters = [
        {'params': [p for n, p in code_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in code_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    query_optimizer_grouped_parameters = [
        {'params': [p for n, p in query_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in query_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    code_optimizer = AdamW(code_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    query_optimizer = AdamW(query_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.n_gpu > 1:
        code_model = torch.nn.DataParallel(code_model)
        query_model = torch.nn.DataParallel(query_model)

    if args.local_rank != -1:
        code_model = torch.nn.parallel.DistributedDataParallel(code_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        query_model = torch.nn.parallel.DistributedDataParallel(query_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
        global_step, tr_loss = train(args, train_dataset, code_model, query_model, tokenizer, code_optimizer, query_optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        code_output_dir = os.path.join(args.output_dir, 'code_model')
        query_output_dir = os.path.join(args.output_dir, 'query_model')
        code_model = model_class.from_pretrained(code_output_dir)
        query_model = model_class.from_pretrained(query_output_dir)
        code_model.to(args.device)
        query_model.to(args.device)
        evaluate(args, code_model, query_model, tokenizer, 'test')



if __name__ == "__main__":
    main()