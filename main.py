#!/usr/bin/env python3

import json
import models

import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
from util import load_dataset, make_iter, Params
#encoding=utf-8
from transformers import (
    BartForConditionalGeneration,BartModel, BartConfig,PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
  )
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import torch
from torch.utils.data import random_split

import os

logs_base_dir = "./logs"
#os.makedirs(logs_base_dir, exist_ok=True)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=16)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=50)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/unsup_train.json')
parser.add_argument('-val_dir',type=str,default='data/unsup_valid.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=1)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/unsup_test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=15)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 
    
def eval(net,model,vocab,data_iter,criterion, epoch):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        input, features,targets,doc_lens, _ = vocab.make_features(batch)
        input, features,targets = Variable(input),Variable(features), Variable(targets.float())
        if use_gpu:
            input = input.cuda()
            features = features.cuda()
            targets = targets.cuda()

        # autoencoder
        encoder_output = model.base_model.encoder(input, return_dict=True).last_hidden_state
        # encoder_output = encoder_output.cuda()

        probs = net(features,doc_lens,encoder_output)
        loss = criterion(probs,targets)
        #writer.add_scalar("Loss/Valid", loss.data, epoch * len(data_iter) + i)
        total_loss += loss.data
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')

    params = Params('config/params.json')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    vocab = utils.Vocab()

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)


    """
    ae model 가져오기 
    """
    #model = BartForConditionalGeneration.from_pretrained('./model/')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')

    if use_gpu:
        model.cuda()
    model.eval()


    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args,embed)
    if use_gpu:
        net.cuda()

    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    #print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))

    print(use_gpu)
    
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    t1 = time()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1,args.epochs+1):
            print("epoch: ", epoch)
            for i,batch in enumerate(train_iter):
                net.train()
                input,features,targets,doc_lens,_ = vocab.make_features(batch)
                input,features,targets = Variable(input),Variable(features), Variable(targets.float())
                if use_gpu:
                    input = input.cuda()
                    features = features.cuda()
                    targets = targets.cuda()

                # autoencoder
                encoder_output = model.base_model.encoder(input, return_dict=True).last_hidden_state
                #print(encoder_output)
                #encoder_output = encoder_output.cuda()
                #print(encoder_output.shape)
                #tmp = list()
                #for eo in encoder_output:
                #    tmp.append(eo)

                probs = net(features,doc_lens,encoder_output)

                #try:
                loss = criterion(probs,targets)
                optimizer.zero_grad()
                loss.backward()
                writer.add_scalar("Loss/Train", loss.data, epoch * len(train_iter) + i)
                clip_grad_norm(net.parameters(), args.max_norm)
                optimizer.step()
                if args.debug:
                    print('Batch ID:%d Loss:%f' %(i,loss.data))
                    continue
                #if i % args.report_every == 0:
                cur_loss = eval(net,model,vocab,val_iter,criterion, epoch)
                writer.add_scalar("Loss/Valid", cur_loss.data, epoch * len(train_iter) + i)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.save()
                print('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
                #except ValueError:
                #    continue
                writer.flush()
    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))
    writer.close()

def test():
     
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    params = Params('config/params.json')
    vocab = utils.Vocab()

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    """
    ae model 가져오기 
    """
    model = BartForConditionalGeneration.from_pretrained('./model/')
    if use_gpu:
        model = model.cuda()
    model.eval()

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        _, features, targets, doc_lens, summaries = vocab.make_features(batch)
        features, targets = Variable(features), Variable(targets.float())
        t1 = time()
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        # autoencoder
        encoder_output = model.base_model.encoder(features, return_dict=True).last_hidden_state
        # encoder_output = encoder_output.cuda()
        tmp = list()
        for eo in encoder_output:
            tmp.append(eo)

        probs = net(features, doc_lens, tmp)

        t2 = time()
        time_cost += t2 - t1
        start = 0

        for doc_id,doc_len in enumerate(doc_lens):
            print("id: ", doc_id)
            ref = summaries[doc_id]
            print(ref)
            stop = start + doc_len
            try:
                prob = probs[start:stop]
            except IndexError:
                continue
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()

            doc = batch['documents'][doc_id].split('\n')[:doc_len]
            labels = [str(l) for l in sorted(list(topk_indices)[:3])]
            print(labels)
            hyp = "\n".join(labels)
            #hyp = doc[topk_indices[0]]

            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write(hyp)
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict():
    logging.info('Loading vocab, pred dataset.Wait a second,please')

    params = Params('config/params.json')
    vocab = utils.Vocab()

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    pred_dataset = utils.Dataset(examples)


    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

    """
    ae
    """
    # load
    model = BartForConditionalGeneration.from_pretrained('./model/')
    if use_gpu:
        model.cuda()
    model.eval()

    """
    extract 모델
    """
    if use_gpu:
        checkpoint = torch.load('checkpoints/RNN_RNN_seed_1.pt')
    else:
        checkpoint = torch.load('checkpoints/RNN_RNN_seed_1.pt', map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    #if not use_gpu:
    #   checkpoint['args'].device = None
    print(checkpoint['args'].device)
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])

    if use_gpu:
        net.cuda()
    net.eval()


    """
    predict
    """
    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        input,features, doc_lens = vocab.make_predict_features(batch)
        input,features= Variable(input), Variable(features)
        t1 = time()
        if use_gpu:
            features = features.cuda()
            input = input.cuda()
        encoder_output = model.base_model.encoder(input, return_dict=True).last_hidden_state

        tmp = list()
        for eo in encoder_output:
            tmp.append(eo)

        probs = net(features, doc_lens, tmp)

        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]

            topk = 1
            #topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()




            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    if args.test:
        test()
    # python main.py -batch_size 1 -predict -load_dir checkpoints/RNN_RNN_seed_1.pt
    elif args.predict:
        predict()
    else:
        train()
