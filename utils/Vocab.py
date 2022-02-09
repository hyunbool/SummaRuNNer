import torch
from transformers import (
    BartForConditionalGeneration,BartModel, BartConfig,PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
  )
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import torch
from torch.utils.data import random_split

class Vocab():
    def __init__(self):
        #self.embed = embed
        #self.word2id = word2id
        #self.id2word = {v:k for k,v in word2id.items()}
        #assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 3
        self.UNK_IDX = 1
        self.SOS_IDX = 0
        self.EOS_IDX = 1
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

    
    def make_features(self,batch,sent_trunc=50,doc_trunc=100,split_token='\n'):
        input_list, sents_list,targets,doc_lens = [],[],[],[]
        # trunc document
        for input, doc,label in zip(batch['input'], batch['documents'],batch['labels']):
            sents = doc.split(split_token)
            labels = label.split("\n")
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))
            input_list.append(input)

        """
        doc
        """
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = self.tokenizer.encode(sent)
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        
        features = []
        for sent in batch_sents:
            feature = sent + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)

        """
        input
        """
        # trunc or pad sent
        max_sent_len = 0
        batch_input = []
        for sent in input_list:
            words = self.tokenizer.encode(sent)
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_input.append(words)

        input_features = []
        for sent in batch_input:
            input_feature = sent+ [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            input_features.append(input_feature)


        input_features = torch.LongTensor(input_features)
        features = torch.LongTensor(features)    
        targets = torch.LongTensor(targets)

        summaries = batch['input']


        return input_features,features,targets,doc_lens, summaries

    def make_predict_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        input_list, sents_list, doc_lens = [],[],[]
        for input, doc in zip(batch['input'], batch['documents']):
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))
            input_list.append(input)

        """
        doc
        """
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = self.tokenizer.encode(sent)
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = sent + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)


        """
        input
        """
        # trunc or pad sent
        max_sent_len = 0
        batch_input = []
        for sent in input_list:
            words = self.tokenizer.encode(sent)
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_input.append(words)

        input_features = []
        for sent in batch_input:
            input_feature = sent + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            input_features.append(input_feature)

        input_features = torch.LongTensor(input_features)
        features = torch.LongTensor(features)

        return input_features, features, doc_lens