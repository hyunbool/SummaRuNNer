from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RNN'
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V, P_D)
        self.rel_pos_embed = nn.Embedding(S, P_D)
        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
            input_size=D,
            hidden_size=H,
            batch_first=True,
            bidirectional=True
        )
        self.sent_RNN = nn.GRU(
            input_size=2 * H,
            hidden_size=H,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(768, 32)
        self.layer_norm = nn.LayerNorm(2 * H, eps=1e-6)
        self.fc = nn.Linear(2 * H, 2 * H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2 * H, 1, bias=False)
        self.salience = nn.Bilinear(2 * H, 2 * H, 1, bias=False)
        self.abs_pos = nn.Linear(P_D, 1, bias=False)

        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def max_pool1d(self, x, seq_lens):
        # x:[N,L,O_in]
        out = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_lens, encoder_output):
        encoder_output = torch.FloatTensor(encoder_output)
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)  # (N,L,D)

        # word level GRU
        H = self.args.hidden_size
        x = self.word_RNN(x)[0]  # (N,2*H,L)
        # word_out = self.avg_pool1d(x,sent_lens)
        word_out = self.max_pool1d(x, sent_lens)
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out, doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]  # (B,max_doc_len,2*H)

        docs = list()
        for i in range(len(encoder_output)):
            docs.append(encoder_output[i].sum(dim=0).tolist())


        docs = torch.FloatTensor(docs)
        docs = torch.sum(encoder_output, 1)

        if self.args.device is not None:
            docs = docs.cuda()

        probs = []

        docs = self.linear(docs)

        # index: batch_num
        for index, doc_len in enumerate(doc_lens):

            valid_hidden = sent_out[index, :doc_len, :]  # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)  # pool

            s = Variable(torch.zeros(1, 2 * H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)  # (1,2*H)
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                # classification layer
                salience = self.salience(h, doc)

                prob = F.sigmoid(salience + self.bias)
                probs.append(prob)

        return torch.cat(probs).squeeze()