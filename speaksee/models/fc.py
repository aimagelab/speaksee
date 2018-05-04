from __future__ import division
from __future__ import absolute_import
import torch
from torch import nn
from torch import distributions
from torch.autograd import Variable
import torch.nn.functional as F
from .CaptioningModel import CaptioningModel

class LSTMCell(nn.Module):
    def __init__(self, opt):
        super(LSTMCell, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.hidden_size = opt.rnn_size
        self.dropout_prob = opt.dropout_prob_lm

        self.i2h = nn.Linear(self.input_encoding_size, 5*self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, 5*self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.i2h.weight)
        nn.init.orthogonal(self.h2h.weight)
        nn.init.constant(self.i2h.bias, 0)
        nn.init.constant(self.h2h.bias, 0)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0])
        sigmoid_chunk = F.sigmoid(all_input_sums[:, :3*self.hidden_size])
        it, ft, ot = sigmoid_chunk.split(self.hidden_size, 1)
        maxout = torch.max(all_input_sums[:,3*self.hidden_size:4*self.hidden_size], all_input_sums[:,4*self.hidden_size:])
        ct = it*maxout + ft * state[1]
        ht = ot * F.tanh(ct)
        ht = self.dropout(ht)

        output = ht.unsqueeze(1)
        state = (ht, ct)
        return output, state

class FC(CaptioningModel):
    def __init__(self, opt):
        super(FC, self).__init__()
        self.vocabulary_size = opt.vocabulary_size
        self.img_feat_size = opt.img_feat_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size

        self.embed = nn.Embedding(self.vocabulary_size, self.input_encoding_size)
        self.fc_image = nn.Linear(self.img_feat_size, self.input_encoding_size)
        self.lstm_cell = LSTMCell(opt)
        self.out_fc = nn.Linear(self.rnn_size, self.vocabulary_size)

        self.ss_prob = .1
        self.init_weights()

    def init_weights(self):
        init_range = .1
        nn.init.uniform(self.embed.weight, -init_range, init_range)
        nn.init.uniform(self.out_fc.weight, -init_range, init_range)
        nn.init.constant(self.out_fc.bias, 0)

    def init_state(self, b_s):
        h0 = Variable(torch.zeros((b_s, self.rnn_size)))
        c0 = Variable(torch.zeros((b_s, self.rnn_size)))
        return (h0, c0)

    def forward(self, images, seq):
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s)
        outputs = []

        for t in range(seq_len):
            if t == 0:
                xt = self.fc_image(images)
            else:
                if t >= 2 and self.ss_prob < 1:
                    # Scheduled sampling
                    coin = images.data.new(b_s).uniform_(0, 1)
                    coin = (coin < self.ss_prob).long() # if 1, true, else sample
                    distr = distributions.Categorical(logits = outputs[-1].squeeze(1))
                    action = distr.sample()
                    it = coin * seq[:, t-1].data + (1-coin) * action.data
                    it = Variable(it, requires_grad=False)
                    if images.is_cuda:
                        it = it.cuda()
                else:
                    it = seq[:, t-1]
                xt = self.embed(it)

            out, state = self.lstm_cell(xt, state)
            out = F.log_softmax(self.out_fc(out), dim=-1)
            outputs.append(out)

        return torch.cat(outputs, 1)
