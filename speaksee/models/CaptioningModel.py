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

        seq_logprob = .0

        outputs = [[] for _ in range(b_s)]
        logprobs = [[] for _ in range(b_s)]
        tmp_outputs = [[[] for __ in range(beam_size)] for _ in range(b_s)]
        selected_words = None

        for t in range(seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            word_logprob, state = self.step(t, state, selected_words, images, None, *args, mode='feedback')
            seq_logprob = seq_logprob + word_logprob.view(b_s, cur_beam_size, -1)

            # Remove sequence if it reaches EOS
            if t > 0:
                mask = selected_words.view(b_s, cur_beam_size, -1) == eos_idx
                seq_logprob = (1-mask).float()*seq_logprob - 999*mask.float()
            selected_logprob, selected_idx = torch.sort(seq_logprob.view(b_s, -1), -1, descending=True)
            selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]

            selected_beam = selected_idx / seq_logprob.shape[-1]
            selected_words = selected_idx - selected_beam*seq_logprob.shape[-1]

            # Update outputs with sequences that reached EOS
            for i in range(b_s):
                outputs[i].extend([tmp_outputs[i][x.item()] for x in torch.masked_select(selected_beam[i], selected_words[i] == eos_idx)])
                logprobs[i].extend([x.item() for x in torch.masked_select(selected_logprob[i], selected_words[i] == eos_idx)])
                tmp_outputs[i] = [tmp_outputs[i][x.item()] for x in selected_beam[i]]
                tmp_outputs[i] = [o+[selected_words[i, x].item(),] for x, o in enumerate(tmp_outputs[i])]

            state = tuple(torch.gather(s.view(b_s, cur_beam_size, -1), 1, selected_beam.unsqueeze(-1).expand(b_s, beam_size, s.shape[-1])).view(-1, s.shape[-1]) for s in state)
            images_exp_shape = (b_s, cur_beam_size) + images_shape[1:]
            images_red_shape = (b_s * beam_size, ) + images_shape[1:]
            selected_beam_red_size = (b_s, beam_size) + tuple(1 for _ in range(len(images_exp_shape)-2))
            selected_beam_exp_size = (b_s, beam_size) + images_exp_shape[2:]
            images_exp = images.view(images_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            images = torch.gather(images_exp, 1, selected_beam_exp).view(images_red_shape)
            seq_logprob = selected_logprob.unsqueeze(-1)
            selected_words = selected_words.view(-1)

        # Update outputs with sequences that did not reach EOS
        for i in range(b_s):
            outputs[i].extend(tmp_outputs[i])
            logprobs[i].extend([x.item() for x in selected_logprob[i]])

            # Sort result
            outputs[i] = [x for _,x in sorted(zip(logprobs[i],outputs[i]), reverse=True)][:out_size]
            if len(outputs[i]) == 1:
                outputs[i] = outputs[i][0]

        return outputs
