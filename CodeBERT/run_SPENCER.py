import argparse
import glob
import logging
import os
import random
import math
import torch.nn.functional as F
import sys


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

def evaluate(args, code_model, query_model, tokenizer, test_data_file)

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
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        accs1, accs3, accs5 = [],[],[]
        code_reprs, summary_reprs = [], []
        n_processed = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            distillated_model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                code_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
                summary_inputs = {'input_ids': batch[2],
                            'attention_mask': batch[3]}
                code_outputs = model(code_inputs)
                summary_outputs = distillated_model(summary_inputs)
                code_norm = F.normalize(code_outputs, dim = 1)
                summary_norm = F.normalize(summary_outputs, dim = 1)
                code_reprs.append(code_norm.cpu())
                summary_reprs.append(summary_norm.cpu())
                n_processed += batch[0].size(0)
        code_reprs, summary_reprs = torch.vstack(code_reprs), torch.vstack(summary_reprs)
        
    num_pool = n_processed // 1000
    accs1, accs3, accs5, mrrs = [], [], [], []
    
    
    file_dir = './results/{}'.format(language)

    for i in tqdm(range(num_pool)):
        code_pool = code_vecs[i*1000:(i+1)*1000]
        file = '{}_batch_result.txt'.format(i)
        with open(os.path.join(file_dir, file), encoding='utf-8') as f:
            batched_data = chunked(f.readlines(), 1000)
            
        for j, batch_data in enumerate(batched_data): # for i in range(pool_size):
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
    parser.add_argument("--code_model_dir", default=None, type=str, required=True,
                        help="The directory where the code model is.")
    parser.add_argument("--query_model_dir", default=None, type=str, required=True,
                        help="The directory where the query model is.")
    parser.add_argument("--top_k", default=None, type=int, required=True,
                        help="The recall number in dual encoder.")    
    parser.add_argument("--test_data_file", default=None, type=str, required=True,
                        help="The input test data file (a txt file).")
    
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

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

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
    code_model = RobertaModel.from_pretrained(args.code_model)
    query_model = RobertaModel.from_pretrained(args.query_model)

    logger.info("Training/evaluation parameters %s", args)
    
    code_model.to(args.device)
    query_model.to(args.device)
    if args.n_gpu > 1:
        code_model = torch.nn.DataParallel(code_model)  
        query_model = torch.nn.DataParallel(query_model)
      

    result = evaluate(args, code_model, query_model, tokenizer, args.test_data_file)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],3)))
            

if __name__ == "__main__":
    main()