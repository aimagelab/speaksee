import torch
from torch import nn
from torch import distributions


class CaptioningModel(nn.Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def init_state(self, b_s, device):
        raise NotImplementedError

    def step(self, t, state, prev_output, images, seq, *args, mode='teacher_forcing'):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    def test(self, images, seq_len, *args):
        device = images.device
        b_s = images.size(0)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, None, *args, mode='feedback')
            out = torch.max(out, -1)[1]
            outputs.append(out)

        return torch.cat([o.unsqueeze(1) for o in outputs], 1)

    def sample_rl(self, images, seq_len, *args):
        device = images.device
        b_s = images.size(0)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        log_probs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, None, *args, mode='feedback')
            distr = distributions.Categorical(logits=out)
            out = distr.sample()
            outputs.append(out)
            log_probs.append(distr.log_prob(out))

        return torch.cat([o.unsqueeze(1) for o in outputs], 1), torch.cat([o.unsqueeze(1) for o in log_probs], 1)

    def beam_search(self, images, seq_len, eos_idx, beam_size, out_size=1, *args):
        device = images.device
        b_s = images.size(0)
        images_shape = images.shape
        state = self.init_state(b_s, device)

        seq_mask = images.data.new_ones((b_s, beam_size, 1))
        seq_logprob = images.data.new_zeros((b_s, 1, 1))
        outputs = []
        log_probs = []
        selected_words = None

        for t in range(seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            word_logprob, state = self.step(t, state, selected_words, images, None, *args, mode='feedback')
            old_seq_logprob = seq_logprob
            word_logprob = word_logprob.view(b_s, cur_beam_size, -1)
            seq_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(b_s, cur_beam_size) != eos_idx).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = old_seq_logprob.expand_as(seq_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                seq_logprob = seq_mask*seq_logprob + old_seq_logprob*(1-seq_mask)

            selected_logprob, selected_idx = torch.sort(seq_logprob.view(b_s, -1), -1, descending=True)
            selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]

            selected_beam = selected_idx / seq_logprob.shape[-1]
            selected_words = selected_idx - selected_beam*seq_logprob.shape[-1]

            new_state = []
            for s in state:
                shape = [int(sh) for sh in s.shape]
                s = torch.gather(s.view(*([b_s, cur_beam_size] + shape[1:])), 1, selected_beam.unsqueeze(-1).expand(
                    *([b_s, beam_size] + shape[1:])))
                s = s.view(*([-1,] + shape[1:]))
                new_state.append(s)
            state = tuple(new_state)

            images_exp_shape = (b_s, cur_beam_size) + images_shape[1:]
            images_red_shape = (b_s * beam_size, ) + images_shape[1:]
            selected_beam_red_size = (b_s, beam_size) + tuple(1 for _ in range(len(images_exp_shape)-2))
            selected_beam_exp_size = (b_s, beam_size) + images_exp_shape[2:]
            images_exp = images.view(images_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            images = torch.gather(images_exp, 1, selected_beam_exp).view(images_red_shape)
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))

            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1, selected_beam.unsqueeze(-1).expand(b_s, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(b_s, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(b_s, beam_size, seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(b_s, beam_size, seq_len))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)
        return outputs, log_probs
