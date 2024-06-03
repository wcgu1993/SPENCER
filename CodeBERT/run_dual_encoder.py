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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, optimizer):
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
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()
    last_results ={'acc1': 0,  'acc3': 0, 'acc5': 0}
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            code_inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            summary_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
            code_outputs = model(code_inputs)
            summary_outputs = model(summary_inputs)
            
            aug_code_outputs = model(code_inputs)
            aug_summary_outputs = model(nl_inputs)

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
            

            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel trainingexi
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
           
            train_iterator.set_postfix(loss=loss.item(), acc1=last_results['acc1'], acc3=last_results['acc3'], acc5=last_results['acc5'])


            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info('loss %s', str(tr_loss - logging_loss))
                        last_results = results
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break


        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, code_model, summary_model, tokenizer)

            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(last_output_dir)
            logger.info("Saving model checkpoint to %s", last_output_dir)
            idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            with open(idx_file, 'w', encoding='utf-8') as idxf:
                idxf.write(str(args.start_epoch + idx) + '\n')

            optimizer_last = os.path.join(last_output_dir, 'optimizer.pt')
            torch.save('optimizer': optimizer.state_dict(), optimizer_last)
            scheduler_last = os.path.join(last_output_dir, 'scheduler.pt')           
            torch.save('code_scheduler': cheduler.state_dict(), scheduler_last)
            logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

            step_file = os.path.join(last_output_dir, 'step_file.txt')
            with open(step_file, 'w', encoding='utf-8') as stepf:
                stepf.write(str(global_step) + '\n')

            if (results['acc1'] > best_acc):
                best_acc = results['acc1']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", output_dir)

                optimizer_last = os.path.join(last_output_dir, 'optimizer.pt')
                torch.save('optimizer': optimizer.state_dict(), optimizer_last)
                scheduler_last = os.path.join(last_output_dir, 'scheduler.pt')
                torch.save('scheduler':scheduler.state_dict(), scheduler_last)
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

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

 


def evaluate(args, model, tokenizer, prefix=""):
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
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                code_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
                summary_inputs = {'input_ids': batch[2],
                            'attention_mask': batch[3]}
                code_outputs = model(code_inputs)
                summary_outputs = model(summary_inputs)
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
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10000,
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
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
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

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # args.task_name = args.task_name.lower()
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in code_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in code_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(code_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(checkpoint_last)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(optimizer_file['code_optimizer'])

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(code_model)


    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, code_optimizer, summary_optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best')
        model = model_class.from_pretrained(output_dir)
        model.to(args.device)
        evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
