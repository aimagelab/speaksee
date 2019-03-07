from __future__ import division
from __future__ import absolute_import
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from .CaptioningModel import CaptioningModel


class BottomupTopdownAttention(CaptioningModel):
    def __init__(self, vocab_size, bos_idx, with_relu=False, with_visual_sentinel=False, det_feat_size=2048,
                 input_encoding_size=1000, rnn_size=1000, att_size=512, ss_prob=.0):
        super(BottomupTopdownAttention, self).__init__()
        self.vocab_size = vocab_size
        self.bos_idx = bos_idx
        self.with_relu = with_relu
        self.with_visual_sentinel = with_visual_sentinel
        self.det_feat_size = det_feat_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.att_size = att_size

        self.embed = nn.Embedding(vocab_size, input_encoding_size)
        self.lstm_cell_1 = nn.LSTMCell(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        self.lstm_cell_2 = nn.LSTMCell(rnn_size + det_feat_size, rnn_size)
        self.att_va = nn.Linear(det_feat_size, att_size, bias=False)
        self.att_ha = nn.Linear(rnn_size, att_size, bias=False)
        self.att_a = nn.Linear(att_size, 1, bias=False)

        if self.with_visual_sentinel:
            self.W_sx = nn.Linear(det_feat_size + rnn_size + input_encoding_size, rnn_size)
            self.W_sh = nn.Linear(rnn_size, rnn_size)
            self.W_sah = nn.Linear(rnn_size, att_size, bias=False)
            self.W_sas = nn.Linear(rnn_size, att_size, bias=False)
            self.W_sa = nn.Linear(att_size, 1, bias=False)
            self.fc_sentinel = nn.Linear(rnn_size, det_feat_size)

        self.out_fc = nn.Linear(rnn_size, vocab_size)

        self.ss_prob = ss_prob
        self.init_weights()

    def init_weights(self):
        std = 0.00999999977648
        nn.init.normal_(self.embed.weight, std=std)
        nn.init.normal_(self.out_fc.weight, std=std)
        nn.init.constant_(self.out_fc.bias, 0)
        nn.init.normal_(self.att_va.weight, std=std)
        nn.init.normal_(self.att_ha.weight, std=std)
        nn.init.normal_(self.att_a.weight, std=std)
        nn.init.normal_(self.lstm_cell_1.weight_ih, std=std)
        nn.init.normal_(self.lstm_cell_1.weight_hh, std=std)
        nn.init.constant_(self.lstm_cell_1.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_1.bias_ih[self.rnn_size:2*self.rnn_size], 1)
        nn.init.constant_(self.lstm_cell_1.bias_hh, 0)
        nn.init.constant_(self.lstm_cell_1.bias_hh[self.rnn_size:2*self.rnn_size], 1)
        nn.init.normal_(self.lstm_cell_2.weight_ih, std=std)
        nn.init.normal_(self.lstm_cell_2.weight_hh, std=std)
        nn.init.constant_(self.lstm_cell_2.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_2.bias_ih[self.rnn_size:2 * self.rnn_size], 1)
        nn.init.constant_(self.lstm_cell_2.bias_hh, 0)
        nn.init.constant_(self.lstm_cell_2.bias_hh[self.rnn_size:2 * self.rnn_size], 1)

        if self.with_visual_sentinel:
            nn.init.normal_(self.W_sx.weight, std=std)
            nn.init.constant_(self.W_sx.bias, 0)
            nn.init.normal_(self.W_sh.weight, std=std)
            nn.init.constant_(self.W_sh.bias, 0)
            nn.init.normal_(self.W_sah.weight, std=std)
            nn.init.normal_(self.W_sas.weight, std=std)
            nn.init.normal_(self.W_sa.weight, std=std)
            nn.init.normal_(self.fc_sentinel.weight, std=std)
            nn.init.constant_(self.fc_sentinel.bias, 0)

    def init_state(self, b_s, device):
        h0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        h0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        return h0_1, c0_1, h0_2, c0_2

    def step(self, t, state, prev_output, detections, seq, *args, mode='teacher_forcing'):
        assert (mode in ['teacher_forcing', 'feedback'])
        device = detections.device
        b_s = detections.size(0)
        bos_idx = self.bos_idx
        state_1, state_2 = state[:2], state[2:]
        detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
        detections_mean = torch.sum(detections, 1) / torch.sum(detections_mask, 1)

        if mode == 'teacher_forcing':
            if self.training and t > 0 and self.ss_prob > .0:
                # Scheduled sampling
                coin = detections.data.new(b_s).uniform_(0, 1)
                coin = (coin < self.ss_prob).long()
                distr = distributions.Categorical(logits=prev_output)
                action = distr.sample()
                it = coin * action.data + (1 - coin) * seq[:, t - 1].data
                it = it.to(device)
            else:
                it = seq[:, t]
        elif mode == 'feedback': # test
            if t == 0:
                it = detections.data.new_full((b_s,), bos_idx).long()
            else:
                it = prev_output

        xt = self.embed(it)
        if self.with_relu:
            xt = F.relu(xt)
        input_1 = torch.cat([state_2[0], detections_mean, xt], 1)

        if self.with_visual_sentinel:
            g_t = torch.sigmoid(self.W_sx(input_1) + self.W_sh(state_1[0]))
        state_1 = self.lstm_cell_1(input_1, state_1)

        att_weights = torch.tanh(self.att_va(detections) + self.att_ha(state_1[0]).unsqueeze(1))
        att_weights = self.att_a(att_weights)

        if self.with_visual_sentinel:
            s_t = g_t * torch.tanh(state_1[1])
            fc_sentinel = self.fc_sentinel(s_t).unsqueeze(1)
            if self.with_relu:
                fc_sentinel = F.relu(fc_sentinel)
            detections = torch.cat([fc_sentinel, detections], 1)
            detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
            sent_att_weights = torch.tanh(self.W_sas(s_t) + self.att_ha(state_1[0])).unsqueeze(1)
            sent_att_weights = self.W_sa(sent_att_weights)
            att_weights = torch.cat([sent_att_weights, att_weights], 1)

        att_weights = F.softmax(att_weights, 1)
        att_weights = detections_mask * att_weights
        att_weights = att_weights / torch.sum(att_weights, 1, keepdim=True)
        att_detections = torch.sum(detections * att_weights, 1)
        input_2 = torch.cat([state_1[0], att_detections], 1)

        state_2 = self.lstm_cell_2(input_2, state_2)
        out = F.log_softmax(self.out_fc(state_2[0]), dim=-1)
        return out, (state_1[0], state_1[1], state_2[0], state_2[1])
