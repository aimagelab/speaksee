from __future__ import division
from __future__ import absolute_import
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
from .CaptioningModel import CaptioningModel


class BottomupTopdownAttention_Sentinel(CaptioningModel):
    def __init__(self, vocab_size, det_feat_size=2048, input_encoding_size=1000, rnn_size=1000, att_size=512, ss_prob=.0):
        super(BottomupTopdownAttention_Sentinel, self).__init__()
        self.vocab_size = vocab_size
        self.det_feat_size = det_feat_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.att_size = att_size

        self.embed = nn.Embedding(vocab_size, input_encoding_size)

        self.W1_is = nn.Linear(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        self.W1_hs = nn.Linear(rnn_size, rnn_size)

        self.lstm_cell_1 = nn.LSTMCell(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        self.lstm_cell_2 = nn.LSTMCell(rnn_size + det_feat_size, rnn_size)

        self.att_va = nn.Linear(det_feat_size, att_size, bias=False)
        self.att_ha = nn.Linear(rnn_size, att_size, bias=False)
        self.att_a = nn.Linear(att_size, 1, bias=False)

        self.att_s = nn.Linear(det_feat_size, att_size, bias=False)

        self.out_fc = nn.Linear(rnn_size, vocab_size)
        self.s_fc = nn.Linear(rnn_size, det_feat_size)

        self.ss_prob = ss_prob
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.out_fc.weight)
        nn.init.constant_(self.out_fc.bias, 0)

        nn.init.xavier_normal_(self.att_va.weight)
        nn.init.xavier_normal_(self.att_ha.weight)
        nn.init.xavier_normal_(self.att_a.weight)

        nn.init.xavier_normal_(self.att_s.weight)

        nn.init.xavier_normal_(self.s_fc.weight)
        nn.init.constant_(self.s_fc.bias, 0)

        nn.init.xavier_normal_(self.lstm_cell_1.weight_ih, 0)
        nn.init.orthogonal_(self.lstm_cell_1.weight_hh)
        nn.init.constant_(self.lstm_cell_1.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_1.bias_hh, 0)

        nn.init.xavier_normal_(self.lstm_cell_2.weight_ih, 0)
        nn.init.orthogonal_(self.lstm_cell_2.weight_hh)
        nn.init.constant_(self.lstm_cell_2.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_2.bias_hh, 0)

    def init_state(self, b_s, device):
        h0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        h0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        return (h0_1, c0_1), (h0_2, c0_2)

    def forward(self, detections, seq):
        device = detections.device
        b_s = detections.size(0)
        seq_len = seq.size(1)
        detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
        detections_mean = torch.sum(detections, 1) / torch.sum(detections_mask, 1)

        state_1, state_2 = self.init_state(b_s, device)
        outputs = []

        for t in range(seq_len):
            if self.training and t > 0 and self.ss_prob > .0:
                # Scheduled sampling
                coin = detections.data.new(b_s).uniform_(0, 1)
                coin = (coin < self.ss_prob).long()
                distr = distributions.Categorical(logits=outputs[-1].squeeze(1))
                action = distr.sample()
                it = coin * action.data + (1-coin) * seq[:, t-1].data
                it = it.to(device)
            else:
                it = seq[:, t]
            xt = self.embed(it)

            input_1 = torch.cat([state_2[0], detections_mean, xt], 1)
            s_gate = F.sigmoid(self.W1_is(input_1) + self.W1_hs(state_1[0]))
            state_1 = self.lstm_cell_1(input_1, state_1)

            s_t = s_gate * F.tanh(state_1[1])
            s_t = F.relu(self.s_fc(s_t))

            det_weights = F.tanh(self.att_va(detections) + self.att_ha(state_1[0]).unsqueeze(1))
            det_weights = self.att_a(det_weights)
            det_weights = (1-detections_mask)*-9e9 + detections_mask*det_weights

            sent_weights = F.tanh(self.att_s(s_t) + self.att_ha(state_1[0]))
            sent_weights = self.att_a(sent_weights).unsqueeze(1)

            att_weights = F.softmax(torch.cat([det_weights, sent_weights], 1), 1)
            regions = torch.cat([detections, s_t.unsqueeze(1)], 1)

            att_detections = torch.sum(regions * att_weights, 1)
            input_2 = torch.cat([state_1[0], att_detections], 1)

            state_2 = self.lstm_cell_2(input_2, state_2)
            out = F.log_softmax(self.out_fc(state_2[0]), dim=-1)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, 1)

    def test(self, detections, seq_len, bos_idx):
        device = detections.device
        b_s = detections.size(0)
        detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
        detections_mean = torch.sum(detections, 1) / torch.sum(detections_mask, 1)

        state_1, state_2 = self.init_state(b_s, device)
        outputs = []
        probas = []

        for t in range(seq_len):
            if t == 0:
                it = detections.data.new_full((b_s,), bos_idx).long()
            else:
                it = torch.max(outputs[-1].squeeze(1), -1)[1]
            xt = self.embed(it)

            input_1 = torch.cat([state_2[0], detections_mean, xt], 1)
            s_gate = F.sigmoid(self.W1_is(input_1) + self.W1_hs(state_1[0]))
            state_1 = self.lstm_cell_1(input_1, state_1)

            s_t = s_gate * F.tanh(state_1[1])
            s_t = F.relu(self.s_fc(s_t))

            det_weights = F.tanh(self.att_va(detections) + self.att_ha(state_1[0]).unsqueeze(1))
            det_weights = self.att_a(det_weights)
            det_weights = (1 - detections_mask) * -9e9 + detections_mask * det_weights

            sent_weights = F.tanh(self.att_s(s_t) + self.att_ha(state_1[0]))
            sent_weights = self.att_a(sent_weights).unsqueeze(1)

            att_weights = F.softmax(torch.cat([det_weights, sent_weights], 1), 1)
            probas.append(att_weights[:, -1].unsqueeze(1))
            regions = torch.cat([detections, s_t.unsqueeze(1)], 1)

            att_detections = torch.sum(regions * att_weights, 1)
            input_2 = torch.cat([state_1[0], att_detections], 1)

            state_2 = self.lstm_cell_2(input_2, state_2)
            out = F.log_softmax(self.out_fc(state_2[0]), dim=-1)
            outputs.append(out.unsqueeze(1))

        return torch.max(torch.cat(outputs, 1), -1)[1], torch.cat(probas, 1)