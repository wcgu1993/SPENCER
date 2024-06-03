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
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

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
       
            
        for example in data:
            self.examples.append(convert_examples_to_features(example,tokenizer,args))

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,       
                 nl_tokens,
                 nl_ids
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids   
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

        
        
def convert_examples_to_features(example,tokenizer,args):
    
    code_tokens = tokenizer.tokenize(examples.code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length=args.code_length-len(code_ids)

    code_ids+=[tokenizer.pad_token_id]*padding_length    
    nl_tokens=tokenizer.tokenize(examples.nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            with open(input_file, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip().split('<CODESPLIT>')
                    if len(line) != 2:
                        continue
                    code= line[0]
                    nl = line[1]
                    data.append(InputExample(code=code, nl=nl))
            for example in data:
                self.examples.append(convert_examples_to_features(example,tokenizer,args))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))              
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):               
        
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(self.examples[item].nl_ids))
            

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
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(distillated_model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    distillated_model.zero_grad()
    code_model.eval()
    query_model.eval()
    distillated_model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            # code_vec = model(code_inputs=code_inputs)
            # nl_vec = model(nl_inputs=nl_inputs)
            
            # #calculate scores and loss
            # scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))

            with torch.no_grad():
                code_outputs = code_model(code_inputs=code_inputs)
                summary_outputs = query_model(nl_inputs=nl_inputs)
            
            distillated_summary_outputs = distillated_model(nl_inputs=nl_inputs)
            
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

    test_dataset = TextDataset(tokenizer, args, file_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)
     
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(test_dataset))
    logger.info("  Num codes = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    distillated_model.eval()
    code_vecs = [] 
    nl_vecs = []
    n_processed = 0
    for batch in test_dataloader:
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs) 
            nl_vec = distillated_model(nl_inputs=nl_inputs) 
            code_vec = F.normalize(code_vec, dim = 1)
            nl_vec = F.normalize(nl_vec, dim = 1)
            nl_vecs.append(nl_vec.cpu().numpy()) 
            code_vecs.append(code_vec.cpu().numpy()) 
            n_processed += batch[0].size(0)


    model.train()    
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
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result

                        
                        
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