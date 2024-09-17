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
from more_itertools import chunked

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

def ACC(real,predict):
    sum=0.0
    for val in real:
        try: index=predict.index(val)
        except ValueError: index=-1
        if index!=-1: sum=sum+1  
    return sum/float(len(real))

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


def train(args, code_model, query_model, distillated_model, optimizer, train_dataset, tokenizer, saved_dir):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

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

    torch.set_printoptions(profile="full")

    tr_loss, logging_loss = 0.0, 0.0
    distillated_model.zero_grad()
    tr_num = 0
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    distillated_model.train()

    for idx in range(args.num_train_epochs):
        tr_loss = 0.0
        for step,batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            code_inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            summary_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
            #get code and nl vectors
            # code_vec = model(code_inputs=code_inputs)
            # nl_vec = model(nl_inputs=nl_inputs)
            
            # #calculate scores and loss
            # scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))

            with torch.no_grad():
                code_outputs = code_model(**code_inputs)
                summary_outputs = query_model(**summary_inputs)
                code_outputs = code_outputs.pooler_output
                summary_outputs = summary_outputs.pooler_output
            
            distillated_summary_outputs = distillated_model(**summary_inputs)
            distillated_summary_outputs = distillated_summary_outputs.pooler_output

            code_outputs = F.normalize(code_outputs, dim = 1)
            summary_outputs = F.normalize(summary_outputs, dim = 1)
            distillated_summary_outputs = F.normalize(distillated_summary_outputs, dim = 1)
            
            dual_similarity = torch.einsum("ab,cb->ac",code_outputs,summary_outputs)

            distillated_dual_similarity = torch.einsum("ab,cb->ac", code_outputs, distillated_summary_outputs)
            distillated_query_similarity = torch.sum(torch.mul(summary_outputs, distillated_summary_outputs), dim=1)
            query_similarity = torch.ones_like(distillated_query_similarity)
            
            loss = torch.abs(dual_similarity - distillated_dual_similarity).sum() + torch.abs(query_similarity - distillated_query_similarity).sum()
            

            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(distillated_model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, code_model, distillated_model, tokenizer)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          
                     
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)                        
            distillated_model.save_pretrained(saved_dir) 
            logger.info("Saving model checkpoint to %s", saved_dir)
        
        return distillated_model


def evaluate(args, code_model, query_model, tokenizer):

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        accs1, accs3, accs5 = [],[],[]
        code_reprs, summary_reprs = [], []
        n_processed = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            query_model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                code_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
                summary_inputs = {'input_ids': batch[2],
                            'attention_mask': batch[3]}
                code_outputs = code_model(**code_inputs).pooler_output
                summary_outputs = query_model(**summary_inputs).pooler_output
                code_norm = F.normalize(code_outputs, dim = 1)
                summary_norm = F.normalize(summary_outputs, dim = 1)
                code_reprs.append(code_norm.cpu())
                summary_reprs.append(summary_norm.cpu())
                n_processed += batch[0].size(0)
        code_reprs, summary_reprs = torch.vstack(code_reprs), torch.vstack(summary_reprs)
        
    num_pool = n_processed // 1000
    accs1, accs3, accs5, mrrs = [], [], [], []
    
    
    file_dir = './results/{}'.format(args.language)

    for i in tqdm(range(num_pool)):
        code_pool = code_reprs[i*1000:(i+1)*1000]
        file = 'batch_{}.txt'.format(i)
        with open(os.path.join(file_dir, file), encoding='utf-8') as f:
            batched_data = chunked(f.readlines(), 1000)
            
        for j, batch_data in enumerate(batched_data): # for i in range(pool_size):
            desc_vec = np.expand_dims(summary_reprs[i*1000+j], axis=0) # [1 x dim] 
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
    parser.add_argument("--language", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--top_k", default=None, type=int, required=True,
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

    parser.add_argument("--reduce_layer_num", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
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

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)


    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
        last_layer_num = 12
        teacher_layer_num = 12
        ta_layer_num = teacher_layer_num - args.reduce_layer_num
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        code_model = RobertaModel.from_pretrained(os.path.join(args.output_dir, 'code_model'))
        query_model = RobertaModel.from_pretrained(os.path.join(args.output_dir, 'query_model'))
        config.num_hidden_layers = max(last_layer_num - args.reduce_layer_num, 1)
        distillated_model = RobertaModel.from_pretrained(os.path.join(args.output_dir, 'query_model'), config=config)
        logger.info("Training/evaluation parameters %s", args)
    
        code_model.to(args.device)
        query_model.to(args.device)
        distillated_model.to(args.device)
        if args.n_gpu > 1:
            code_model = torch.nn.DataParallel(code_model)  
            query_model = torch.nn.DataParallel(query_model)
            distillated_model = torch.nn.DataParallel(distillated_model)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in distillated_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in distillated_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
   
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        saved_dir = os.path.join(args.output_dir, "query_model_{}_to_{}_layer".format(last_layer_num, config.num_hidden_layers))
        ta_model = train(args, code_model, query_model, distillated_model, optimizer, train_dataset, tokenizer, saved_dir)
        teacher_model = query_model
        results = evaluate(args, code_model, query_model, tokenizer)
        original_mrr = results['mrr']
        distillated_results = evaluate(args, code_model, query_model, tokenizer)
        distillated_mrr = distillated_results['mrr']
        best_distillation_dir = os.path.join(args.output_dir, 'query_model')

        if distillated_mrr / original_mrr < 0.01:
            best_distillation_dir = saved_dir

        last_layer_num = config.num_hidden_layers

        while config.num_hidden_layers > 1 and distillated_mrr / original_mrr < 0.01:
            config.num_hidden_layers = max(last_layer_num - args.reduce_layer_num, 1)
            distillated_model = RobertaModel.from_pretrained(os.path.join(args.output_dir, 'query_model'), config=config)
            distillated_model.to(args.device)

            optimizer_grouped_parameters = [
                {'params': [p for n, p in distillated_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
                {'params': [p for n, p in distillated_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
   
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            
            teacher_saved_dir = os.path.join(args.output_dir, "query_model_{}_to_{}_layer".format(teacher_layer_num, config.num_hidden_layers))
            teacher_distillated_model = train(args, code_model, teacher_model, distillated_model, optimizer, train_dataset, tokenizer, teacher_saved_dir)

            ta_saved_dir = os.path.join(args.output_dir, "query_model_{}_to_{}_layer".format(ta_layer_num, config.num_hidden_layers))
            ta_distillated_model = train(args, code_model, ta_model, distillated_model, optimizer, train_dataset, tokenizer, ta_saved_dir)
            teacher_results = evaluate(args, code_model, teacher_distillated_model, tokenizer)
            teacher_mrr = teacher_results['mrrs']
            ta_results = evaluate(args, code_model, ta_distillated_model, tokenizer)
            ta_mrr = ta_results['mrr']
            if teacher_mrr > ta_mrr:
                if teacher_mrr / original_mrr:
                    best_distillation_dir = teacher_saved_dir
                ta_model = teacher_distillated_model
                ta_layer_num = config.num_hidden_layers
                distillated_mrr = teacher_mrr
            else:
                if ta_mrr / original_mrr:
                    best_distillation_dir = ta_saved_dir
                teacher_model = ta_model
                teacher_layer_num = ta_layer_num
                ta_model = ta_distillated_model
                ta_layer_num = config.num_hidden_layers
                distillated_mrr = ta_mrr

            last_layer_num = config.num_hidden_layers
        
        with open(os.path.join(args.output_dir, 'best-distillation.txt'), 'w') as f:
            f.write(best_distillation_dir)

    if args.do_eval:
        code_model = RobertaModel.from_pretrained(os.path.join(args.output_dir, 'code_model'))
        with open(os.path.join(args.output_dir, 'best-distillation.txt'), 'r') as f:
            query_model_dir = f.readline().strip()
        query_model = RobertaModel.from_pretrained(query_model_dir)
        code_model.to(args.device)
        query_model.to(args.device)        
        result = evaluate(args, code_model, query_model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            

if __name__ == "__main__":
    main()