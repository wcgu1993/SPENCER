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


logger = logging.getLogger(__name__)

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


def train(args, code_model, query_model, distillated_model, tokenizer):
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


    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(scheduler_last['scheduler'], torch.load(scheduler_last))


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

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    distillated_model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    distillated_model.train()
    last_results ={'acc1': 0,  'acc3': 0, 'acc5': 0}
    for idx, _ in enumerate(train_iterator):
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
                code_outputs = code_model(code_inputs)
                summary_outputs = query_model(summary_inputs)
                code_outputs = code_outputs.pooler_output
                summary_outputs = summary_outputs.pooler_output
            
            distillated_summary_outputs = distillated_model(summary_inputs)
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
        results = evaluate(args, code_model, distillated_model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            model_to_save.save_pretrained(output_dir) 
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, distillated_model, tokenizer,file_name,eval_when_training=False):

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
     
       
        code_pool, desc_pool = code_reprs, summary_reprs 
        for i in tqdm(range(n_processed)): # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0) # [1 x dim] 
            sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
            negsims=np.negative(sims)
            predict1 = np.argpartition(negsims, kth=0)#predict=np.argsort(negsims)#
            predict3 = np.argpartition(negsims, kth=4)#predict=np.argsort(negsims)#
            predict10 = np.argpartition(negsims, kth=9)#predict=np.argsort(negsims)#
            predict1 = predict1[:1]   
            predict1 = [int(k) for k in predict1]
            predict3 = predict5[:3]   
            predict3 = [int(k) for k in predict3]
            predict5 = predict5[:5]   
            predict5 = [int(k) for k in predict5]
            real = [i]
            accs1.append(ACC(real,predict1))
            accs3.append(ACC(real,predict3))
            accs5.append(ACC(real,predict5))
            # ndcgs.append(NDCG(real,predict))                     
        
        
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            logger.info('acc1: %f, acc3: %f, acc5: %f,', np.mean(accs1), np.mean(accs3), np.mean(accs5))
            writer.write('acc1: %f\n' % np.mean(accs1))

    return {'acc1':np.mean(accs1), 'acc3': np.mean(accs3), 'acc5': np.mean(accs5)}  

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--code_model_dir", default=None, type=str, required=True,
                        help="The directory where the code model is.")
    parser.add_argument("--query_model_dir", default=None, type=str, required=True,
                        help="The directory where the query model is.")
    parser.add_argument("--distillated_model_dir", default=None, type=str, required=True,
                        help="The directory where the distillated model is.")
    parser.add_argument("--target_layer_num", default=None, type=int, required=True,
                        help="Target layer number for model distillation.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
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

    checkpoint_prefix = 'checkpoint-best'
    code_model_dir = os.path.join(args.code_model_dir, '{}'.format(checkpoint_prefix))  
    code_model = RobertaModel.from_pretrained(code_model_dir) 
    code_model.to(args.device)

    
    query_model_dir = os.path.join(query_model_dir, '{}'.format(checkpoint_prefix))  
    query_model = RobertaModel.from_pretrained(query_model_dir)
    query_model.to(args.device)

    distillated_model_config = RobertaConfig.from_pretrained(args.model_name_or_path)
    distillated_model_config.num_hidden_layers = args.target_layer_num
    distillated_model = RobertaModel.from_pretrained(args.model_name_or_path, config = distillated_model_config) 
 
    logger.info("Training/evaluation parameters %s", args)
    
    distillated_model.to(args.device)
    if args.n_gpu > 1:
        distillated_model = torch.nn.DataParallel(distillated_model)  
            
    # Training
    if args.do_train:
        train(args, code_model, query_model, distillated_model, tokenizer)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best'
            distillated_model_dir = os.path.join(args.distillated_model_dir, '{}'.format(checkpoint_prefix))  
            distillated_model = RobertaModel.from_pretrained(distillated_model_dir) 
        distillated_model.to(args.device)
        result = evaluate(args, code_model, distillated_model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))

    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best'
            distillated_model_dir = os.path.join(args.distillated_model_dir, '{}'.format(checkpoint_prefix))  
            distillated_model = RobertaModel.from_pretrained(distillated_model_dir) 
        distillated_model.to(args.device)
        result = evaluate(args, code_model, distillated_model, tokenizer, args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    main()