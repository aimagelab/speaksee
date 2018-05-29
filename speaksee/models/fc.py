from __future__ import division
from __future__ import absolute_import
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from .CaptioningModel import CaptioningModel

class LSTMCell(nn.Module):
    def __init__(self, input_encoding_size, rnn_size, dropout_prob_lm):
        super(LSTMCell, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.hidden_size = rnn_size
        self.dropout_prob = dropout_prob_lm

        self.i2h = nn.Linear(self.input_encoding_size, 5*self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, 5*self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)

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
    def __init__(self, vocab_size, img_feat_size, input_encoding_size, rnn_size, dropout_prob_lm, ss_prob=.0):
        super(FC, self).__init__()
        self.vocab_size = vocab_size
        self.img_feat_size = img_feat_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size

        self.embed = nn.Embedding(vocab_size, input_encoding_size)
        self.fc_image = nn.Linear(img_feat_size, input_encoding_size)
        self.lstm_cell = LSTMCell(input_encoding_size, rnn_size, dropout_prob_lm)
        self.out_fc = nn.Linear(rnn_size, vocab_size)

        self.ss_prob = ss_prob
        self.init_weights()

    def init_weights(self):
        init_range = .1
        nn.init.uniform_(self.embed.weight, -init_range, init_range)
        nn.init.uniform_(self.out_fc.weight, -init_range, init_range)
        nn.init.constant_(self.out_fc.bias, 0)

    def init_state(self, b_s, device):
        h0 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        return h0, c0

    def forward(self, images, seq):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        outputs = []

        for t in range(seq_len):
            if t == 0:
                xt = self.fc_image(images)
            else:
                if self.training and self.ss_prob > .0:
                    # Scheduled sampling
                    coin = images.data.new(b_s).uniform_(0, 1)
                    coin = (coin < self.ss_prob).long()
                    distr = distributions.Categorical(logits = outputs[-1].squeeze(1))
                    action = distr.sample()
                    it = coin * action.data + (1-coin) * seq[:, t-1].data
                    it = it.to(device)
                else:
                    it = seq[:, t-1]
                xt = self.embed(it)

            out, state = self.lstm_cell(xt, state)
            out = F.log_softmax(self.out_fc(out), dim=-1)
            outputs.append(out)

        return torch.cat(outputs, 1)

    def test(self, images, seq_len):
        device = images.device
        b_s = images.size(0)
        state = self.init_state(b_s, device)
        outputs = []

        for t in range(seq_len):
            if t == 0:
                xt = self.fc_image(images)
            else:
                it = torch.max(outputs[-1].squeeze(1), -1)[1]
                xt = self.embed(it)

            out, state = self.lstm_cell(xt, state)
            out = F.log_softmax(self.out_fc(out), dim=-1)
            outputs.append(out)

        return torch.max(torch.cat(outputs, 1), -1)[1]
