#!/usr/bin/env python3

import json
import models
from models.encoder import Encoder
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

import os
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
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/train_ex.json')
parser.add_argument('-val_dir',type=str,default='data/val_ex.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test_ex.json')
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
    
def eval(net,model,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        input, features,targets,doc_lens = vocab.make_features(batch)
        input, features,targets = Variable(input),Variable(features), Variable(targets.float())
        if use_gpu:
            input = input.cuda()
            features = features.cuda()
            targets = targets.cuda()

        # autoencoder
        encoder_output = model(input)

        probs = net(features,doc_lens,encoder_output)
        loss = criterion(probs,targets)
        total_loss += loss.data
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    params = Params('config/params.json')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)


    """
    ae model 가져오기 
    """
    # select model and load trained model


    # load
    pretrained_dict = torch.load("./entire_ae.pt")
    model = Encoder(params)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    model.to(params.device)
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
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    t1 = time()
    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(1,args.epochs+1):
            for i,batch in enumerate(train_iter):
                input,features,targets,doc_lens = vocab.make_features(batch)
                input,features,targets = Variable(input),Variable(features), Variable(targets.float())
                if use_gpu:
                    input = input.cuda()
                    features = features.cuda()
                    targets = targets.cuda()

                # autoencoder
                encoder_output = model(input)

                probs = net(features,doc_lens,encoder_output)

                try:
                    loss = criterion(probs,targets)
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm(net.parameters(), args.max_norm)
                    optimizer.step()
                    if args.debug:
                        print('Batch ID:%d Loss:%f' %(i,loss.data))
                        continue
                    if i % args.report_every == 0:
                        cur_loss = eval(net,model,vocab,val_iter,criterion)
                        if cur_loss < min_loss:
                            min_loss = cur_loss
                            best_path = net.save()
                        logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                                % (epoch,min_loss,cur_loss))
                except ValueError:
                    continue
    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))

def test():
     
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

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
        features,_,summaries,doc_lens = vocab.make_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('\n'.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict():
    params = Params('config/params.json')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]

    pred_dataset = utils.Dataset(examples)


    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    """
    extract 모델
    """
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    """
    ae
    """
    # load
    pretrained_dict = torch.load("./entire_ae.pt")
    model = Encoder(params)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    model.to(params.device)
    model.eval()

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
            input = input.cuda()
            features = features.cuda()

            encoder_output = model(features)

            probs = net(features, doc_lens, encoder_output)
        else:
            probs = net(Variable(features), doc_lens, encoder_output)
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




            doc = batch['doc'][doc_id].split('</s> ')[:doc_len]
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
